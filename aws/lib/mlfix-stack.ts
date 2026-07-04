import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as apigw from 'aws-cdk-lib/aws-apigateway';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as logs from 'aws-cdk-lib/aws-logs';
import { Duration, RemovalPolicy } from 'aws-cdk-lib';
import * as path from 'path';

export class MlfixStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // ---------- DynamoDB tables ----------

    // Episodes: one row per fix attempt uploaded from a user.
    // PK = user_id (hash), SK = episode_id (range). On-demand billing = no fixed cost.
    const episodesTable = new dynamodb.Table(this, 'EpisodesTable', {
      tableName: 'mlfix-episodes',
      partitionKey: { name: 'user_id', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'episode_id', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: RemovalPolicy.DESTROY, // dev only — see notes below
      pointInTimeRecovery: false,
    });

    // Global secondary index so we can query by category (for aggregation).
    episodesTable.addGlobalSecondaryIndex({
      indexName: 'category-ts-index',
      partitionKey: { name: 'category', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'ts', type: dynamodb.AttributeType.STRING },
    });

    // Global bandit arms: aggregated Beta(alpha, beta) per (category, model).
    // Users pull this on daemon startup to get a warm start.
    const banditTable = new dynamodb.Table(this, 'BanditTable', {
      tableName: 'mlfix-bandit-global',
      partitionKey: { name: 'category', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'model', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // Prompt registry: versioned specialist prompts. Users pull latest on startup.
    const promptsTable = new dynamodb.Table(this, 'PromptsTable', {
      tableName: 'mlfix-prompts',
      partitionKey: { name: 'category', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'version', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // ---------- S3 replay buffer ----------

    const replayBucket = new s3.Bucket(this, 'ReplayBucket', {
      bucketName: `mlfix-replay-${this.account}-${this.region}`,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true, // dev only — cleans up when stack is destroyed
      lifecycleRules: [{
        id: 'expire-old',
        expiration: Duration.days(30), // auto-delete after 30 days (keeps free tier safe)
      }],
    });

    // ---------- Lambda functions ----------

    const commonEnv = {
      EPISODES_TABLE: episodesTable.tableName,
      BANDIT_TABLE: banditTable.tableName,
      PROMPTS_TABLE: promptsTable.tableName,
      REPLAY_BUCKET: replayBucket.bucketName,
    };

    const commonLambdaProps: Omit<lambda.FunctionProps, 'code' | 'handler'> = {
      runtime: lambda.Runtime.PYTHON_3_11,
      timeout: Duration.seconds(15),
      memorySize: 256,
      environment: commonEnv,
      logRetention: logs.RetentionDays.ONE_WEEK, // keep CloudWatch logs small
    };

    // 1. Ingest Lambda — receives episodes from clients
    const ingestFn = new lambda.Function(this, 'IngestFn', {
      ...commonLambdaProps,
      functionName: 'mlfix-ingest',
      handler: 'ingest.handler',
      code: lambda.Code.fromAsset(path.join(__dirname, '..', 'lambda', 'ingest')),
    });
    episodesTable.grantWriteData(ingestFn);
    replayBucket.grantPut(ingestFn);

    // 2. Sync Lambda — serves global bandit priors + latest prompts
    const syncFn = new lambda.Function(this, 'SyncFn', {
      ...commonLambdaProps,
      functionName: 'mlfix-sync',
      handler: 'sync.handler',
      code: lambda.Code.fromAsset(path.join(__dirname, '..', 'lambda', 'sync')),
    });
    banditTable.grantReadData(syncFn);
    promptsTable.grantReadData(syncFn);

    // 3. Aggregator Lambda — runs nightly, rolls up episodes into bandit stats
    const aggregatorFn = new lambda.Function(this, 'AggregatorFn', {
      ...commonLambdaProps,
      functionName: 'mlfix-aggregator',
      timeout: Duration.minutes(5),
      memorySize: 512,
      handler: 'aggregator.handler',
      code: lambda.Code.fromAsset(path.join(__dirname, '..', 'lambda', 'aggregator')),
    });
    episodesTable.grantReadData(aggregatorFn);
    banditTable.grantReadWriteData(aggregatorFn);

    // Schedule the aggregator: 03:00 UTC daily (08:30 IST)
    new events.Rule(this, 'AggregatorSchedule', {
      schedule: events.Schedule.cron({ minute: '0', hour: '3' }),
      targets: [new targets.LambdaFunction(aggregatorFn)],
    });

    // ---------- API Gateway ----------

    const api = new apigw.RestApi(this, 'MlfixApi', {
      restApiName: 'mlfix-api',
      description: 'mlfix backend API',
      deployOptions: {
        stageName: 'v1',
        throttlingBurstLimit: 20,
        throttlingRateLimit: 10,
      },
      defaultCorsPreflightOptions: {
        allowOrigins: apigw.Cors.ALL_ORIGINS,
        allowMethods: ['GET', 'POST', 'OPTIONS'],
      },
    });

    // API key + usage plan (rate limits, prevents abuse)
    const apiKey = api.addApiKey('MlfixApiKey', {
      apiKeyName: 'mlfix-default-key',
      description: 'Default key for mlfix clients',
    });

    const usagePlan = api.addUsagePlan('MlfixUsagePlan', {
      name: 'mlfix-usage-plan',
      throttle: { burstLimit: 20, rateLimit: 10 },
      quota: { limit: 10000, period: apigw.Period.MONTH },
    });
    usagePlan.addApiKey(apiKey);
    usagePlan.addApiStage({ stage: api.deploymentStage });

    // Routes
    const episodes = api.root.addResource('episodes');
    episodes.addMethod('POST', new apigw.LambdaIntegration(ingestFn), {
      apiKeyRequired: true,
    });

    const sync = api.root.addResource('sync');
    sync.addMethod('GET', new apigw.LambdaIntegration(syncFn), {
      apiKeyRequired: true,
    });

    // ---------- Outputs ----------

    new cdk.CfnOutput(this, 'ApiUrl', { value: api.url });
    new cdk.CfnOutput(this, 'ApiKeyId', { value: apiKey.keyId });
    new cdk.CfnOutput(this, 'ReplayBucketName', { value: replayBucket.bucketName });
  }
}
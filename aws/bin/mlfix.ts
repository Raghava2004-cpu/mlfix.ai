#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { MlfixStack } from '../lib/mlfix-stack';

const app = new cdk.App();

new MlfixStack(app, 'MlfixStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || 'ap-south-1',
  },
  description: 'mlfix backend: episodes ingest, sync, and nightly aggregator',
});
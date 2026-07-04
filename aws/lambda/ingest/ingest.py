"""Receives episodes from mlfix clients. Writes to DynamoDB + S3 raw dump."""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

_ddb = boto3.resource("dynamodb")
_s3 = boto3.client("s3")

EPISODES_TABLE = os.environ["EPISODES_TABLE"]
REPLAY_BUCKET = os.environ["REPLAY_BUCKET"]


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def handler(event, context):
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _response(400, {"error": "invalid JSON"})

    user_id = body.get("user_id")
    episode = body.get("episode")

    if not user_id or not isinstance(episode, dict):
        return _response(400, {"error": "user_id and episode required"})

    episode_id = episode.get("episode_id") or uuid.uuid4().hex[:12]
    ts = episode.get("ts") or datetime.utcnow().isoformat()
    category = episode.get("category", "generic")

    # Truncate large fields before storing (protect free tier)
    def _cap(s, n):
        return (s or "")[:n]

    item = {
        "user_id": user_id,
        "episode_id": episode_id,
        "ts": ts,
        "category": category,
        "model_used": episode.get("model_used", "unknown"),
        "judge_success": bool(episode.get("judge_success", False)),
        "user_decision": episode.get("user_decision"),
        "critic_approved": bool(episode.get("critic_approved", False)),
        "execution_success": episode.get("execution_success"),
        "tokens_in": int(episode.get("tokens_in", 0)),
        "tokens_out": int(episode.get("tokens_out", 0)),
        # Text fields, capped
        "error": _cap(episode.get("error", ""), 2000),
        "explanation": _cap(episode.get("explanation", ""), 1000),
        # Don't store user's raw code in DynamoDB — privacy + size.
        # Only S3 gets it, and only if user opted in (client controls what's sent).
        "code_hash": episode.get("code_hash"),
    }

    # Write to DynamoDB (structured, queryable)
    _ddb.Table(EPISODES_TABLE).put_item(Item=item)

    # Write to S3 (full raw episode, for offline analysis)
    day = ts[:10]  # YYYY-MM-DD
    key = f"episodes/{day}/{user_id}/{episode_id}.json"
    _s3.put_object(
        Bucket=REPLAY_BUCKET,
        Key=key,
        Body=json.dumps(episode).encode("utf-8"),
        ContentType="application/json",
    )

    log.info("ingested user=%s episode=%s category=%s", user_id, episode_id, category)
    return _response(200, {"ok": True, "episode_id": episode_id})
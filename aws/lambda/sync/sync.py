"""Serves global bandit priors + prompt templates to clients."""
from __future__ import annotations

import json
import logging
import os
from decimal import Decimal

import boto3

log = logging.getLogger()
log.setLevel(logging.INFO)

_ddb = boto3.resource("dynamodb")

BANDIT_TABLE = os.environ["BANDIT_TABLE"]
PROMPTS_TABLE = os.environ["PROMPTS_TABLE"]


def _to_json_safe(obj):
    """DynamoDB returns Decimal; JSON can't serialize it."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    return obj


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(_to_json_safe(body)),
    }


def handler(event, context):
    try:
        bandit = _ddb.Table(BANDIT_TABLE).scan().get("Items", [])
    except Exception as e:
        log.exception("bandit scan failed")
        bandit = []

    try:
        prompts = _ddb.Table(PROMPTS_TABLE).scan().get("Items", [])
    except Exception as e:
        log.exception("prompts scan failed")
        prompts = []

    # Group prompts by category, pick highest version
    latest_prompts: dict[str, dict] = {}
    for p in prompts:
        cat = p.get("category")
        if not cat:
            continue
        if cat not in latest_prompts or p.get("version", "") > latest_prompts[cat].get("version", ""):
            latest_prompts[cat] = p

    return _response(200, {
        "bandit_arms": bandit,
        "prompts": list(latest_prompts.values()),
    })
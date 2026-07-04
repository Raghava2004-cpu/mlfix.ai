"""Nightly job: rolls up episodes into global bandit arms."""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta

import boto3
from boto3.dynamodb.conditions import Key

log = logging.getLogger()
log.setLevel(logging.INFO)

_ddb = boto3.resource("dynamodb")

EPISODES_TABLE = os.environ["EPISODES_TABLE"]
BANDIT_TABLE = os.environ["BANDIT_TABLE"]


def _reward(item: dict) -> float:
    """Compute reward from an episode. Same shape as the daemon uses."""
    # Prefer user decision; fall back to judge if user didn't decide.
    dec = item.get("user_decision")
    if dec == "accept":
        return 1.0
    if dec == "reject":
        return 0.0
    return 1.0 if item.get("judge_success") else 0.0


def handler(event, context):
    # Look at the last 24h using the category-ts-index. We scan through
    # every category the naive way for now (small data). Real system: iterate
    # categories from a known list.
    since = (datetime.utcnow() - timedelta(days=1)).isoformat()

    # Aggregation: (category, model) -> (successes, failures)
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])

    categories = [
        "shape_mismatch", "cuda_oom", "dependency", "data_leakage",
        "training_dynamics", "syntax", "import", "type", "generic",
    ]

    table = _ddb.Table(EPISODES_TABLE)

    for cat in categories:
        resp = table.query(
            IndexName="category-ts-index",
            KeyConditionExpression=Key("category").eq(cat) & Key("ts").gte(since),
        )
        for item in resp.get("Items", []):
            model = item.get("model_used", "unknown")
            r = _reward(item)
            if r >= 0.5:
                counts[(cat, model)][0] += 1
            else:
                counts[(cat, model)][1] += 1

    # Write into bandit table. We store alpha=successes+1, beta=failures+1.
    bandit = _ddb.Table(BANDIT_TABLE)
    for (cat, model), (wins, losses) in counts.items():
        bandit.put_item(Item={
            "category": cat,
            "model": model,
            "alpha": wins + 1,
            "beta": losses + 1,
            "updated_at": datetime.utcnow().isoformat(),
            "sample_size": wins + losses,
        })
        log.info("bandit updated: cat=%s model=%s wins=%d losses=%d",
                 cat, model, wins, losses)

    return {"ok": True, "arms_updated": len(counts)}
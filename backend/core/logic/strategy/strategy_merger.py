"""Helpers for merging strategist outputs with bureau data."""

from __future__ import annotations

import re
from typing import Any, Mapping, MutableMapping, Optional

from backend.core.logic.compliance.constants import (
    FallbackReason,
    StrategistFailureReason,
    normalize_action_tag,
)
from backend.core.logic.strategy.fallback_manager import determine_fallback_action
from backend.core.logic.utils.names_normalization import normalize_creditor_name
from backend.core.models.account import Account
from backend.core.models.strategy import StrategyPlan


def merge_strategy_outputs(
    strategy_obj: StrategyPlan,
    bureau_data_obj: MutableMapping[str, MutableMapping[str, list[Account]]],
) -> None:
    """Align strategist ``strategy_obj`` with ``bureau_data_obj``."""

    if not isinstance(strategy_obj, StrategyPlan):
        raise TypeError("strategy_obj must be a StrategyPlan instance")
    plan = strategy_obj

    def norm_key(name: str, number: str) -> tuple[str, str]:
        norm_name = normalize_creditor_name(name)
        digits = re.sub(r"\D", "", number or "")
        last4 = digits[-4:] if digits else ""
        return norm_name, last4

    index = {}
    for item in plan.accounts:
        key = norm_key(item.name, item.account_number or "")
        index[key] = item

    for payload in bureau_data_obj.values():
        for items in payload.values():
            if not isinstance(items, list):
                continue
            for acc in items:
                if not isinstance(acc, Account):
                    raise TypeError("bureau_data_obj must contain Account instances")

                key = norm_key(acc.name, acc.account_number or "")
                src = index.get(key)
                raw_action: Optional[str] = None
                if src is None:
                    acc.extras["strategist_failure_reason"] = (
                        StrategistFailureReason.MISSING_INPUT
                    )
                    continue
                rec = src.recommendation
                raw_action = (
                    rec.recommended_action if rec else None
                ) or acc.extras.get("recommendation")
                tag, action = normalize_action_tag(raw_action)
                if raw_action is None:
                    acc.extras["strategist_failure_reason"] = (
                        StrategistFailureReason.EMPTY_OUTPUT
                    )
                elif raw_action and not tag:
                    acc.extras["strategist_failure_reason"] = (
                        StrategistFailureReason.UNRECOGNIZED_FORMAT
                    )
                    acc.extras["fallback_unrecognized_action"] = True
                if tag:
                    acc.extras["action_tag"] = tag
                    acc.extras["recommended_action"] = action
                elif raw_action:
                    acc.extras["recommended_action"] = raw_action

                acc.extras["strategist_raw_action"] = raw_action
                if rec and rec.advisor_comment:
                    acc.extras["advisor_comment"] = rec.advisor_comment
                if rec and rec.flags:
                    acc.extras["flags"] = rec.flags


def handle_strategy_fallbacks(
    bureau_data_obj: MutableMapping[str, Any],
    classification_map: Mapping[str, Any],
    audit=None,
    log_list: list | None = None,
) -> None:
    """Apply fallback logic and log strategy decisions."""

    for bureau, payload in bureau_data_obj.items():
        for section, items in payload.items():
            if not isinstance(items, list):
                continue
            for acc in items:
                raw_action = acc.get("strategist_raw_action")
                tag = acc.get("action_tag")
                failure_reason = acc.get("strategist_failure_reason")
                acc_id = acc.get("account_id") or acc.get("name")

                if failure_reason and audit:
                    audit.log_account(
                        acc_id,
                        {
                            "stage": "strategist_failure",
                            "failure_reason": failure_reason.value,
                            **(
                                {"raw_action": raw_action}
                                if (
                                    failure_reason
                                    == StrategistFailureReason.UNRECOGNIZED_FORMAT
                                    and raw_action
                                )
                                else {}
                            ),
                        },
                    )
                if (
                    failure_reason == StrategistFailureReason.MISSING_INPUT
                    and log_list is not None
                ):
                    log_list.append(
                        f"[{bureau}] No strategist entry for '{acc.get('name')}' ({acc.get('account_number')})"
                    )
                if (
                    failure_reason == StrategistFailureReason.UNRECOGNIZED_FORMAT
                    and raw_action
                ):
                    print(
                        f"[âš ï¸] Unrecognised strategist action '{raw_action}' for {acc.get('name')}"
                    )

                if not tag:
                    strategist_action = raw_action if raw_action else None
                    if raw_action is None:
                        fallback_reason = FallbackReason.NO_RECOMMENDATION
                    else:
                        raw_key = str(raw_action).strip().lower().replace(" ", "_")
                        fallback_reason = (
                            FallbackReason.KEYWORD_MATCH
                            if raw_key == FallbackReason.KEYWORD_MATCH.value
                            else FallbackReason.UNRECOGNIZED_TAG
                        )

                    fallback_action = determine_fallback_action(acc)
                    keywords_trigger = fallback_action == "dispute"

                    if keywords_trigger:
                        acc["action_tag"] = "dispute"
                        if raw_action:
                            acc["recommended_action"] = "Dispute"
                        else:
                            acc.setdefault("recommended_action", "Dispute")

                        if log_list is not None and (raw_action is None or not tag):
                            if raw_action:
                                log_list.append(
                                    f"[{bureau}] Fallback dispute overriding '{raw_action}' for '{acc.get('name')}' ({acc.get('account_number')})"
                                )
                            else:
                                log_list.append(
                                    f"[{bureau}] Fallback dispute (no recommendation) for '{acc.get('name')}' ({acc.get('account_number')})"
                                )
                    else:
                        if log_list is not None and (raw_action is None or not tag):
                            log_list.append(
                                f"[{bureau}] Evaluated fallback for '{acc.get('name')}' ({acc.get('account_number')})"
                            )

                    overrode_strategist = bool(raw_action) and bool(keywords_trigger)

                    if audit:
                        audit.log_account(
                            acc_id,
                            {
                                "stage": "strategy_fallback",
                                "fallback_reason": fallback_reason.value,
                                "strategist_action": strategist_action,
                                **(
                                    {"raw_action": strategist_action}
                                    if acc.get("fallback_unrecognized_action")
                                    and strategist_action
                                    else {}
                                ),
                                "overrode_strategist": overrode_strategist,
                                **(
                                    {"failure_reason": failure_reason.value}
                                    if failure_reason
                                    else {}
                                ),
                            },
                        )
                if audit:
                    cls = classification_map.get(str(acc.get("account_id")))
                    audit.log_account(
                        acc_id,
                        {
                            "stage": "strategy_decision",
                            "action": acc.get("action_tag") or None,
                            "recommended_action": acc.get("recommended_action"),
                            "flags": acc.get("flags"),
                            "reason": acc.get("advisor_comment")
                            or acc.get("analysis")
                            or raw_action,
                            "classification": getattr(cls, "classification", cls) or {},
                        },
                    )


def merge_strategy_data(
    strategy_obj: StrategyPlan | dict,
    bureau_data_obj: MutableMapping[str, Any],
    classification_map: Mapping[str, Any],
    audit=None,
    log_list: list | None = None,
) -> None:
    """Wrapper combining merge and fallback handling."""

    plan = (
        strategy_obj
        if isinstance(strategy_obj, StrategyPlan)
        else StrategyPlan.from_dict(strategy_obj)
    )

    for payload in bureau_data_obj.values():
        for section, items in payload.items():
            if isinstance(items, list):
                payload[section] = [
                    Account.from_dict(a) if isinstance(a, dict) else a for a in items
                ]

    merge_strategy_outputs(plan, bureau_data_obj)

    for payload in bureau_data_obj.values():
        for section, items in payload.items():
            if isinstance(items, list):
                payload[section] = [
                    a.to_dict() if isinstance(a, Account) else a for a in items
                ]

    handle_strategy_fallbacks(bureau_data_obj, classification_map, audit, log_list)


__all__ = [
    "merge_strategy_outputs",
    "handle_strategy_fallbacks",
    "merge_strategy_data",
]

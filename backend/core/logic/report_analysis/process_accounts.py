import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, cast

from backend.audit.audit import AuditLogger
from backend.core.logic.compliance.constants import FallbackReason
from backend.core.logic.strategy.fallback_manager import determine_fallback_action
from backend.core.logic.utils.names_normalization import (
    normalize_bureau_name,
    normalize_creditor_name,
)
from backend.core.logic.utils.text_parsing import enforce_collection_status

BUREAUS = ["Experian", "Equifax", "TransUnion"]

logger = logging.getLogger(__name__)


@dataclass
class Account:
    """Simple container for account data."""

    name: str
    bureaus: List[str]
    account_number: Optional[str] = None
    status: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Account":
        """Create an ``Account`` instance from a raw dictionary."""
        name = data.get("name", "")
        bureaus = list(data.get("bureaus", []))
        account_number = data.get("account_number")
        status = data.get("status")
        extra = {
            k: v
            for k, v in data.items()
            if k not in {"name", "bureaus", "account_number", "status"}
        }
        return cls(
            name=name,
            bureaus=bureaus,
            account_number=account_number,
            status=status,
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to a serializable dictionary."""
        data = {
            "name": self.name,
            "bureaus": self.bureaus,
        }
        if self.account_number is not None:
            data["account_number"] = self.account_number
        if self.status is not None:
            data["status"] = self.status
        data.update(self.extra)
        return data

    def get(self, key: str, default: Any = None) -> Any:
        if key in {"name", "bureaus", "account_number", "status"}:
            return getattr(self, key, default)
        return self.extra.get(key, default)

    def __getitem__(self, key: str) -> Any:
        if key in {"name", "bureaus", "account_number", "status"}:
            return getattr(self, key)
        return self.extra[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key in {"name", "bureaus", "account_number", "status"}:
            setattr(self, key, value)
        else:
            self.extra[key] = value


# 💡 Functions to enrich goodwill accounts with hardship context


def infer_hardship_reason(account: Account) -> str:
    """Infer a short hardship reason based on account notes."""
    notes = (account.get("status", "") + " " + account.get("name", "")).lower()
    if "medical" in notes:
        return "I was dealing with a medical emergency that affected my finances"
    elif "job" in notes or "unemploy" in notes:
        return "I experienced an unexpected job loss which disrupted my ability to make payments"
    return (
        "I went through a temporary hardship that affected my ability to stay current"
    )


def infer_personal_impact(account: Account) -> str:
    """Describe how the hardship affected the client."""
    return "difficulty qualifying for a loan and increased financial stress"


def infer_recovery_summary(account: Account) -> str:
    """Summarize how the client has recovered from the hardship."""
    return "Since then, I have taken full responsibility and made all payments on time"


# 📥 Load analyzed SmartCredit report


def load_analyzed_report(json_path: Path) -> Mapping[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return cast(Mapping[str, Any], json.load(f))


# 🎯 Process and categorize by bureau


def process_analyzed_report(
    json_path: str | Path,
    audit: AuditLogger | None = None,
    log_list: list[str] | None = None,
) -> Mapping[str, Dict[str, List[Any]]]:
    """Process a SmartCredit analysis report and categorize accounts by bureau.

    ``log_list`` if provided will be appended with messages describing
    automatic tagging or skipped items.
    """
    data = load_analyzed_report(Path(json_path))

    output: Dict[str, Dict[str, List[Any]]] = {
        bureau: {
            "disputes": [],
            "goodwill": [],
            "inquiries": [],
            "high_utilization": [],
        }
        for bureau in BUREAUS
    }

    seen_entries = set()

    # 1. Negative Accounts
    for account_data in data.get("negative_accounts", []):
        enforce_collection_status(account_data)
        account = Account.from_dict(account_data)
        is_goodwill_closed = account.get("goodwill_on_closed", False)
        name_key = normalize_creditor_name(account.name)
        for bureau in account.bureaus:
            bureau_norm = normalize_bureau_name(bureau)
            identifier = account.get("account_number_last4") or account.get(
                "account_fingerprint"
            )
            key = (name_key, identifier, bureau_norm)
            if bureau_norm in output and key not in seen_entries:
                if is_goodwill_closed:
                    enriched_account = Account.from_dict(account.to_dict())
                    enriched_account["hardship_reason"] = infer_hardship_reason(account)
                    enriched_account["impact"] = infer_personal_impact(account)
                    enriched_account["recovery_summary"] = infer_recovery_summary(
                        account
                    )
                    output[bureau_norm]["goodwill"].append(enriched_account)
                else:
                    output[bureau_norm]["disputes"].append(account)
                seen_entries.add(key)
            elif bureau_norm in output:
                logger.info(
                    "suppressed_account %s",
                    {
                        "suppression_reason": "duplicate_entry",
                        "name": account.name,
                        "bureau": bureau_norm,
                    },
                )
                if log_list is not None:
                    log_list.append(
                        f"[{bureau_norm}] Duplicate entry skipped for '{account.name}'"
                    )

    # 2. Open Accounts with Issues
    for account_data in data.get("open_accounts_with_issues", []):
        enforce_collection_status(account_data)
        account = Account.from_dict(account_data)
        is_goodwill = account.get("goodwill_candidate", False) or account.get(
            "goodwill_on_closed", False
        )
        name_key = normalize_creditor_name(account.name)
        for bureau in account.bureaus:
            bureau_norm = normalize_bureau_name(bureau)
            identifier = account.get("account_number_last4") or account.get(
                "account_fingerprint"
            )
            key = (name_key, identifier, bureau_norm)
            if bureau_norm in output and key not in seen_entries:
                if is_goodwill:
                    enriched_account = Account.from_dict(account.to_dict())
                    enriched_account["hardship_reason"] = infer_hardship_reason(account)
                    enriched_account["impact"] = infer_personal_impact(account)
                    enriched_account["recovery_summary"] = infer_recovery_summary(
                        account
                    )
                    output[bureau_norm]["goodwill"].append(enriched_account)
                else:
                    output[bureau_norm]["disputes"].append(account)
                seen_entries.add(key)
            elif bureau_norm in output:
                logger.info(
                    "suppressed_account %s",
                    {
                        "suppression_reason": "duplicate_entry",
                        "name": account.name,
                        "bureau": bureau_norm,
                    },
                )
                if log_list is not None:
                    log_list.append(
                        f"[{bureau_norm}] Duplicate entry skipped for '{account.name}'"
                    )

    # 3. High Utilization Accounts - NOT sent to disputes
    for account_data in data.get("high_utilization_accounts", []):
        enforce_collection_status(account_data)
        account = Account.from_dict(account_data)
        name_key = normalize_creditor_name(account.name)
        for bureau in account.bureaus:
            bureau_norm = normalize_bureau_name(bureau)
            identifier = account.get("account_number_last4") or account.get(
                "account_fingerprint"
            )
            key = (name_key, identifier, bureau_norm)
            if bureau_norm in output and key not in seen_entries:
                output[bureau_norm]["high_utilization"].append(account)
                seen_entries.add(key)
            elif bureau_norm in output:
                logger.info(
                    "suppressed_account %s",
                    {
                        "suppression_reason": "duplicate_entry",
                        "name": account.name,
                        "bureau": bureau_norm,
                    },
                )
                if log_list is not None:
                    log_list.append(
                        f"[{bureau_norm}] Duplicate entry skipped for '{account.name}'"
                    )

    # 4. Inquiries (exclude matched)
    def norm(name: str | None) -> str:
        return cast(str, normalize_creditor_name(name or ""))

    matched_names = set()
    for match in data.get("account_inquiry_matches", []):
        if isinstance(match, dict):
            matched_names.add(norm(match.get("creditor_name")))

    all_account_names = {
        norm(acc.get("name"))
        for acc in data.get("all_accounts", [])
        if "closed"
        not in str(acc.get("status") or acc.get("account_status") or "").lower()
    }

    for inquiry in data.get("inquiries", []):
        bureau_raw = inquiry.get("bureau")
        bureau = normalize_bureau_name(bureau_raw)
        name = norm(inquiry.get("creditor_name"))
        matched = name in matched_names or name in all_account_names
        if bureau in output and not matched:
            inquiry["bureau"] = bureau
            output[bureau]["inquiries"].append(inquiry)
        elif bureau in output and log_list is not None:
            log_list.append(
                f"[{bureau}] Inquiry '{inquiry.get('creditor_name')}' matched known account and was skipped"
            )

    def apply_fallback_tags(
        data_dict: Dict[str, Dict[str, List[Any]]],
        audit: AuditLogger | None,
        log_list: list[str] | None = None,
    ) -> None:
        """Tag obvious dispute items when the strategist left them blank."""
        for bureau, payload in data_dict.items():
            for sec in ["disputes", "goodwill", "high_utilization"]:
                for acc in payload.get(sec, []):
                    if acc.get("action_tag"):
                        continue
                    action = determine_fallback_action(acc)
                    if action == "dispute":
                        original_tag = acc.get("action_tag")
                        if audit:
                            audit.log_account(
                                acc.get("account_id") or acc.get("name"),
                                {
                                    "stage": "pre_strategy_fallback",
                                    "fallback_reason": FallbackReason.KEYWORD_MATCH.value,
                                    "original_tag": original_tag,
                                },
                            )
                        acc["action_tag"] = "dispute"
                        if acc.get("recommended_action") is None:
                            acc["recommended_action"] = "Dispute"
                        if log_list is not None:
                            log_list.append(
                                f"[{bureau}] Fallback dispute tag applied to '{acc.get('name')}' ({FallbackReason.KEYWORD_MATCH.value})",
                            )

    apply_fallback_tags(output, audit, log_list)
    return output


# 💾 Save separate JSON files per bureau


def save_bureau_outputs(
    output_data: Mapping[str, Dict[str, List[Any]]], output_folder: Path
) -> None:
    """Write bureau data to JSON files."""
    output_folder.mkdir(parents=True, exist_ok=True)
    for bureau, content in output_data.items():
        serializable = {
            section: [
                item.to_dict() if isinstance(item, Account) else item for item in items
            ]
            for section, items in content.items()
        }
        out_path = output_folder / f"{bureau}_payload.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=4, ensure_ascii=False)
        print(f"[💾] Saved: {out_path}")


# ✅ CLI Run

if __name__ == "__main__":
    input_path = Path("uploads/analyzed_report.json")
    output_path = Path("output")

    result = process_analyzed_report(input_path)
    save_bureau_outputs(result, output_path)

    print("\n[✅] Bureau-level segmentation complete.")

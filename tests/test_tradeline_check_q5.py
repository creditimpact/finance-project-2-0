"""Unit tests for Q5 (Ownership & Responsibility Declaration)."""

from __future__ import annotations

from backend.tradeline_check.q5_ownership import evaluate_q5, ALLOWED_FIELDS


def test_q5_allowed_fields():
    """Verify Q5 only reads account_description."""
    assert ALLOWED_FIELDS == {"account_description"}


def test_q5_individual_exact():
    """Test exact match: Individual → ok/individual/1.0"""
    bureau_obj = {"account_description": "Individual"}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    assert result["version"] == "q5_ownership_v1"
    assert result["status"] == "ok"
    assert result["declared_responsibility"] == "individual"
    assert result["signals"] == ["ACCOUNT_DESCRIPTION:INDIVIDUAL"]
    assert result["evidence_fields"] == ["account_description"]
    assert result["evidence"] == {"account_description": "Individual"}
    assert result["explanation"] == "declared from account_description"
    assert result["confidence"] == 1.0


def test_q5_joint_exact():
    """Test exact match: Joint → ok/joint/1.0"""
    bureau_obj = {"account_description": "Joint"}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    assert result["status"] == "ok"
    assert result["declared_responsibility"] == "joint"
    assert result["signals"] == ["ACCOUNT_DESCRIPTION:JOINT"]
    assert result["confidence"] == 1.0


def test_q5_authorized_user_exact():
    """Test exact match: Authorized User → ok/authorized_user/1.0"""
    bureau_obj = {"account_description": "Authorized User"}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    assert result["status"] == "ok"
    assert result["declared_responsibility"] == "authorized_user"
    assert result["signals"] == ["ACCOUNT_DESCRIPTION:AUTHORIZED_USER"]
    assert result["confidence"] == 1.0


def test_q5_authorized_variant():
    """Test variant: Authorized → ok/authorized_user/1.0"""
    bureau_obj = {"account_description": "Authorized"}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    assert result["status"] == "ok"
    assert result["declared_responsibility"] == "authorized_user"
    assert result["signals"] == ["ACCOUNT_DESCRIPTION:AUTHORIZED_USER"]
    assert result["confidence"] == 1.0


def test_q5_authorizeduser_variant():
    """Test variant: authorizeduser → ok/authorized_user/1.0"""
    bureau_obj = {"account_description": "authorizeduser"}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    assert result["status"] == "ok"
    assert result["declared_responsibility"] == "authorized_user"
    assert result["confidence"] == 1.0


def test_q5_cosigner_conservative():
    """Test conservative mapping: Co-signer → ok/joint/0.9"""
    bureau_obj = {"account_description": "Co-signer"}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    assert result["status"] == "ok"
    assert result["declared_responsibility"] == "joint"
    assert result["signals"] == ["ACCOUNT_DESCRIPTION:JOINT"]
    assert result["confidence"] == 0.9


def test_q5_cosigner_no_hyphen():
    """Test conservative mapping: cosigner → ok/joint/0.9"""
    bureau_obj = {"account_description": "cosigner"}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    assert result["status"] == "ok"
    assert result["declared_responsibility"] == "joint"
    assert result["confidence"] == 0.9


def test_q5_terminated_ambiguous():
    """Test ambiguous: Terminated → unknown/unknown/0.5"""
    bureau_obj = {"account_description": "Terminated"}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    assert result["status"] == "unknown"
    assert result["declared_responsibility"] == "unknown"
    assert result["signals"] == []
    assert result["explanation"] == "account_description ambiguous or unrecognized"
    assert result["confidence"] == 0.5


def test_q5_missing_none():
    """Test missing: None → skipped_missing_data"""
    bureau_obj = {}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    assert result["status"] == "skipped_missing_data"
    assert result["declared_responsibility"] == "unknown"
    assert result["signals"] == []
    assert result["explanation"] == "account_description missing"
    assert result["confidence"] == 0.5


def test_q5_placeholder_dash():
    """Test placeholder: '--' → skipped_missing_data"""
    bureau_obj = {"account_description": "--"}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    assert result["status"] == "skipped_missing_data"
    assert result["declared_responsibility"] == "unknown"
    assert result["explanation"] == "account_description missing"


def test_q5_placeholder_na():
    """Test placeholder: 'n/a' → skipped_missing_data"""
    bureau_obj = {"account_description": "n/a"}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    assert result["status"] == "skipped_missing_data"


def test_q5_placeholder_unknown():
    """Test placeholder: 'unknown' → skipped_missing_data"""
    bureau_obj = {"account_description": "unknown"}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    assert result["status"] == "skipped_missing_data"


def test_q5_empty_string():
    """Test empty string → skipped_missing_data"""
    bureau_obj = {"account_description": ""}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    assert result["status"] == "skipped_missing_data"


def test_q5_whitespace_only():
    """Test whitespace only → skipped_missing_data"""
    bureau_obj = {"account_description": "   "}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    assert result["status"] == "skipped_missing_data"


def test_q5_unknown_value():
    """Test unknown string: 'Company Account' → unknown"""
    bureau_obj = {"account_description": "Company Account"}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    assert result["status"] == "unknown"
    assert result["declared_responsibility"] == "unknown"
    assert result["signals"] == []
    assert result["confidence"] == 0.5


def test_q5_case_insensitive():
    """Test case insensitive: 'INDIVIDUAL' → ok/individual"""
    bureau_obj = {"account_description": "INDIVIDUAL"}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    assert result["status"] == "ok"
    assert result["declared_responsibility"] == "individual"
    assert result["confidence"] == 1.0


def test_q5_whitespace_normalization():
    """Test whitespace normalization: '  Joint  ' → ok/joint"""
    bureau_obj = {"account_description": "  Joint  "}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    assert result["status"] == "ok"
    assert result["declared_responsibility"] == "joint"
    assert result["confidence"] == 1.0


def test_q5_bureau_isolation():
    """Test bureau isolation: different inputs yield different outputs"""
    placeholders = {"--", "n/a", "unknown"}

    # Bureau 1: Individual
    bureau1 = {"account_description": "Individual"}
    result1 = evaluate_q5(bureau1, placeholders)
    assert result1["declared_responsibility"] == "individual"

    # Bureau 2: Joint
    bureau2 = {"account_description": "Joint"}
    result2 = evaluate_q5(bureau2, placeholders)
    assert result2["declared_responsibility"] == "joint"

    # Bureau 3: Authorized User
    bureau3 = {"account_description": "Authorized User"}
    result3 = evaluate_q5(bureau3, placeholders)
    assert result3["declared_responsibility"] == "authorized_user"

    # Verify outputs are independent
    assert result1 != result2 != result3


def test_q5_non_blocking_ok():
    """Test non-blocking: Q5 status=ok never blocks"""
    bureau_obj = {"account_description": "Individual"}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    # Q5 never sets findings or blocked_questions
    assert "findings" not in result
    assert "blocked_questions" not in result
    assert result["status"] == "ok"


def test_q5_non_blocking_unknown():
    """Test non-blocking: Q5 status=unknown never blocks"""
    bureau_obj = {"account_description": "Terminated"}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    assert "findings" not in result
    assert "blocked_questions" not in result
    assert result["status"] == "unknown"


def test_q5_non_blocking_skipped():
    """Test non-blocking: Q5 status=skipped never blocks"""
    bureau_obj = {}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    assert "findings" not in result
    assert "blocked_questions" not in result
    assert result["status"] == "skipped_missing_data"


def test_q5_evidence_restricted():
    """Test evidence restricted to allowed fields only"""
    bureau_obj = {
        "account_description": "Individual",
        "account_status": "Open",
        "balance_owed": "$1000",
        "creditor_type": "Bank",
    }
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    # Evidence must only include account_description
    assert result["evidence_fields"] == ["account_description"]
    assert result["evidence"] == {"account_description": "Individual"}
    assert "account_status" not in result["evidence"]
    assert "balance_owed" not in result["evidence"]
    assert "creditor_type" not in result["evidence"]


def test_q5_schema_structure():
    """Test Q5 output schema structure is complete"""
    bureau_obj = {"account_description": "Joint"}
    placeholders = {"--", "n/a", "unknown"}

    result = evaluate_q5(bureau_obj, placeholders)

    # Required keys
    required_keys = {
        "version",
        "status",
        "declared_responsibility",
        "signals",
        "evidence_fields",
        "evidence",
        "explanation",
        "confidence",
    }
    assert set(result.keys()) == required_keys

    # Type checks
    assert isinstance(result["version"], str)
    assert isinstance(result["status"], str)
    assert isinstance(result["declared_responsibility"], str)
    assert isinstance(result["signals"], list)
    assert isinstance(result["evidence_fields"], list)
    assert isinstance(result["evidence"], dict)
    assert isinstance(result["explanation"], str)
    assert isinstance(result["confidence"], float)

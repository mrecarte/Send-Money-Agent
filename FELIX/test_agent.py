"""
Unit tests for the Send Money Agent.

Run:  python test_agent.py
"""

import json
import sys

from models import SessionState, BankDetails
from tools import (
    get_receive_currency, get_fx_rate,
    get_next_missing_field, is_collection_complete,
    apply_update, _resolve_current_phase, _validate_fields,
    process_user_input, format_transfer_receipt,
    submit_transfer, get_transfer_summary,
)



class MockToolContext:
    """Simulates ADK's ToolContext for unit testing without a running session."""
    def __init__(self, state: dict | None = None):
        self.state = state or {}


def _make_complete_session() -> SessionState:
    """Returns a fully-populated SessionState — used by receipt and submission tests."""
    s = SessionState()
    s.transfer.destination_country = "Mexico"
    s.transfer.send_amount          = 500.0
    s.transfer.send_currency        = "USD"
    s.transfer.payout_method        = "bank_deposit"
    s.transfer.receive_currency     = "MXN"
    s.transfer.fx_rate              = 17.15
    s.transfer.fees                 = 7.50
    s.transfer.receive_amount       = 8576.25
    s.payout.bank_details = BankDetails(
        bank_name      = "BBVA Mexico",
        account_number = "1234567890",
        routing_number = "021000021",
    )
    s.sender.legal_name = "John Smith"
    s.sender.phone      = "+15552345678"
    s.sender.email      = "john@email.com"
    s.recipient.legal_name             = "Maria Lopez"
    s.recipient.phone                  = "+525512345678"
    s.recipient.relationship_to_sender = "sister"
    s.compliance.purpose_of_transfer = "Family support"
    s.compliance.source_of_funds     = "Salary"
    return s


# Country & FX helpers

def test_country_to_currency():
    """All 7 supported corridors map to the correct ISO code."""
    assert get_receive_currency("Mexico")             == "MXN"
    assert get_receive_currency("Guatemala")          == "GTQ"
    assert get_receive_currency("Honduras")           == "HNL"
    assert get_receive_currency("Colombia")           == "COP"
    assert get_receive_currency("El Salvador")        == "USD"
    assert get_receive_currency("Dominican Republic") == "DOP"
    assert get_receive_currency("Nicaragua")          == "NIO"
    assert get_receive_currency("India")              is None   # unsupported
    assert get_receive_currency("the Mexico")         == "MXN"  # "The " stripping
    print("PASS  test_country_to_currency")


def test_fx_rate_same_currency():
    """El Salvador sends USD → USD: short-circuits to 1.0 without hitting yfinance."""
    assert get_fx_rate("USD", "USD") == 1.0
    print("PASS  test_fx_rate_same_currency")


# get_next_missing_field (collection order/puzzle behaviour)

def test_next_field_fresh_session():
    """First field requested on an empty session is destination_country."""
    assert get_next_missing_field(SessionState()) == ("transfer", "destination_country")
    print("PASS  test_next_field_fresh_session")


def test_next_field_skips_already_filled():
    """Puzzle behaviour: fields filled out-of-order are skipped correctly."""
    s = SessionState()
    s.transfer.destination_country = "Mexico"
    s.transfer.send_amount = 500.0
    # send_currency should be next, both above are already filled
    assert get_next_missing_field(s) == ("transfer", "send_currency")
    print("PASS  test_next_field_skips_already_filled")


def test_next_field_payout_branch_bank():
    """Bank deposit branch asks for bank_name → account_number → routing_number."""
    s = SessionState()
    s.transfer.destination_country = "Mexico"
    s.transfer.send_amount  = 500.0
    s.transfer.send_currency = "USD"
    s.transfer.payout_method = "bank_deposit"
    assert get_next_missing_field(s) == ("payout.bank_details", "bank_name")
    print("PASS  test_next_field_payout_branch_bank")


def test_next_field_payout_branch_wallet():
    """Mobile wallet branch asks for wallet_provider first."""
    s = SessionState()
    s.transfer.destination_country = "Mexico"
    s.transfer.send_amount   = 500.0
    s.transfer.send_currency  = "USD"
    s.transfer.payout_method  = "mobile_wallet"
    assert get_next_missing_field(s) == ("payout.wallet_details", "wallet_provider")
    print("PASS  test_next_field_payout_branch_wallet")


def test_collection_complete():
    """is_collection_complete returns True only when every required field is filled."""
    assert is_collection_complete(_make_complete_session()) is True
    assert is_collection_complete(SessionState())           is False
    print("PASS  test_collection_complete")


# apply_update

def test_apply_update_two_level():
    """2-level dot-notation paths are written correctly."""
    s = SessionState()
    changed = apply_update(s, {"transfer.send_amount": 300.0, "sender.legal_name": "John"})
    assert s.transfer.send_amount == 300.0
    assert s.sender.legal_name == "John"
    assert set(changed) == {"transfer.send_amount", "sender.legal_name"}
    print("PASS  test_apply_update_two_level")


def test_apply_update_three_level():
    """3-level paths lazily initialise the payout sub-object."""
    s = SessionState()
    apply_update(s, {"payout.bank_details.bank_name": "BBVA Mexico"})
    assert s.payout.bank_details is not None
    assert s.payout.bank_details.bank_name == "BBVA Mexico"
    print("PASS  test_apply_update_three_level")


def test_correction_recorded_in_audit_log():
    """Overwriting a filled field appends a CorrectionEntry; first fill does not."""
    s = SessionState()
    apply_update(s, {"transfer.send_amount": 300.0})
    assert len(s.corrections) == 0            # first fill — not a correction

    apply_update(s, {"transfer.send_amount": 500.0})
    assert len(s.corrections) == 1
    assert s.corrections[0].old_value == 300.0
    assert s.corrections[0].new_value == 500.0
    print("PASS  test_correction_recorded_in_audit_log")


def test_fx_recalculation_on_trigger_field():
    """Changing a FX_TRIGGER_FIELDS value recalculates all derived financials."""
    s = SessionState()
    apply_update(s, {
        "transfer.destination_country": "El Salvador",  # USD corridor
        "transfer.send_amount":         100.0,
        "transfer.send_currency":       "USD",
        "transfer.payout_method":       "bank_deposit",
    })
    # El Salvador: USD→USD, rate = 1.0, fee = 1.5%, receive = 98.5
    assert s.transfer.fx_rate       == 1.0
    assert s.transfer.fees          == 1.5
    assert s.transfer.receive_amount == 98.5
    print("PASS  test_fx_recalculation_on_trigger_field")


# _resolve_current_phase

def test_phase_resolution_all_stages():
    """Phase resolves correctly at every stage of the collection."""
    assert _resolve_current_phase(SessionState()) == "transfer"

    s_people = _make_complete_session()
    s_people.sender.legal_name = None
    assert _resolve_current_phase(s_people) == "people"

    s_compliance = _make_complete_session()
    s_compliance.compliance.purpose_of_transfer = None
    assert _resolve_current_phase(s_compliance) == "compliance"

    assert _resolve_current_phase(_make_complete_session()) == "confirmation"
    print("PASS  test_phase_resolution_all_stages")


# _validate_fields

def test_validate_amount_bounds():
    """Amount must be > 0 and ≤ $10,000."""
    _, e = _validate_fields({"transfer.send_amount": 0})
    assert any("greater than zero" in err for err in e)

    _, e = _validate_fields({"transfer.send_amount": 15_000})
    assert any("10,000" in err for err in e)

    valid, e = _validate_fields({"transfer.send_amount": 500})
    assert "transfer.send_amount" in valid and e == []
    print("PASS  test_validate_amount_bounds")


def test_validate_unsupported_country():
    """Countries outside the 7 supported corridors are rejected with a helpful message."""
    _, e = _validate_fields({"transfer.destination_country": "India"})
    assert any("Felix does not yet support" in err for err in e)

    valid, e = _validate_fields({"transfer.destination_country": "Mexico"})
    assert "transfer.destination_country" in valid and e == []
    print("PASS  test_validate_unsupported_country")


def test_validate_email_format():
    """Malformed email addresses are rejected."""
    _, e = _validate_fields({"sender.email": "notanemail"})
    assert e

    valid, e = _validate_fields({"sender.email": "john@email.com"})
    assert "sender.email" in valid and e == []
    print("PASS  test_validate_email_format")


def test_validate_routing_number():
    """Routing number must be exactly 9 digits; dashes are stripped and stored clean."""
    _, e = _validate_fields({"payout.bank_details.routing_number": "12345"})
    assert e

    valid, e = _validate_fields({"payout.bank_details.routing_number": "021-000-021"})
    assert e == []
    assert valid["payout.bank_details.routing_number"] == "021000021"  # dashes stripped
    print("PASS  test_validate_routing_number")


def test_validate_phone_digits():
    """Phone numbers with fewer than 7 digits are rejected."""
    _, e = _validate_fields({"sender.phone": "123"})
    assert e

    valid, e = _validate_fields({"sender.phone": "+1 555 234 5678"})
    assert "sender.phone" in valid and e == []
    print("PASS  test_validate_phone_digits")


# ADK tools via MockToolContext

def test_process_user_input_multi_field():
    """Multiple fields in one call are all captured and returned in newly_filled."""
    ctx = MockToolContext()
    result = json.loads(process_user_input(
        json.dumps({
            "transfer.destination_country": "Mexico",
            "transfer.send_amount":         500.0,
            "transfer.send_currency":       "USD",
        }),
        ctx,
    ))
    assert len(result["newly_filled"]) == 3
    assert result["corrections"]       == []
    assert result["current_phase"]     == "transfer"
    assert result["next_field_label"]  is not None
    print("PASS  test_process_user_input_multi_field")


def test_process_user_input_correction():
    """Second write to an already-filled field shows up in corrections, not newly_filled."""
    ctx = MockToolContext()
    process_user_input(json.dumps({"transfer.send_amount": 300.0}), ctx)
    result = json.loads(process_user_input(json.dumps({"transfer.send_amount": 500.0}), ctx))
    assert "transfer.send_amount" in result["corrections"]
    assert result["newly_filled"] == []
    print("PASS  test_process_user_input_correction")


def test_process_user_input_validation_error():
    """Invalid fields are rejected and surfaced in validation_errors."""
    ctx = MockToolContext()
    result = json.loads(process_user_input(
        json.dumps({"transfer.send_amount": -50}), ctx
    ))
    assert result["validation_errors"] != []
    # Invalid value must NOT have been written to state
    s = SessionState(**ctx.state["session"])
    assert s.transfer.send_amount is None
    print("PASS  test_process_user_input_validation_error")


def test_receipt_blocked_when_incomplete():
    """format_transfer_receipt returns a NOT READY message if called too early."""
    result = format_transfer_receipt(MockToolContext())
    assert "NOT READY" in result
    print("PASS  test_receipt_blocked_when_incomplete")


def test_receipt_contains_all_sections():
    """Completed session produces a receipt with every required section."""
    ctx = MockToolContext({"session": _make_complete_session().model_dump()})
    receipt = format_transfer_receipt(ctx)
    for expected in [
        "TRANSFER CONFIRMATION", "TXN-",
        "BBVA Mexico", "John Smith", "Maria Lopez",
        "Family support", "SENDER", "RECIPIENT", "COMPLIANCE",
    ]:
        assert expected in receipt, f"Missing: {expected}"
    print("PASS  test_receipt_contains_all_sections")


def test_receipt_shows_change_log():
    """A session with corrections shows a CHANGE LOG section in the receipt."""
    s = _make_complete_session()
    apply_update(s, {"transfer.send_amount": 600.0})   # correction: 500 → 600
    ctx = MockToolContext({"session": s.model_dump()})
    receipt = format_transfer_receipt(ctx)
    assert "CHANGE LOG" in receipt
    assert "500.0" in receipt and "600.0" in receipt
    print("PASS  test_receipt_shows_change_log")


def test_submit_transfer_marks_submitted():
    """submit_transfer sets submitted=True and returns the confirmation number."""
    s = _make_complete_session()
    s.confirmation.confirmation_number = "TXN-20240101-1234"
    ctx = MockToolContext({"session": s.model_dump()})
    result = json.loads(submit_transfer(ctx))

    assert result["submitted"]           is True
    assert result["confirmation_number"] == "TXN-20240101-1234"
    assert SessionState(**ctx.state["session"]).confirmation.submitted is True
    print("PASS  test_submit_transfer_marks_submitted")


def test_get_transfer_summary_strips_none():
    """get_transfer_summary only returns fields that have been filled."""
    s = SessionState()
    s.transfer.destination_country = "Mexico"
    s.transfer.send_amount = 500.0
    ctx = MockToolContext({"session": s.model_dump()})
    summary = json.loads(get_transfer_summary(ctx))

    assert "destination_country" in summary["transfer"]
    assert "send_amount"         in summary["transfer"]
    assert "send_currency"   not in summary["transfer"]   # not yet filled
    assert summary["current_phase"] == "transfer"
    print("PASS  test_get_transfer_summary_strips_none")


if __name__ == "__main__":
    tests = [
        test_country_to_currency,
        test_fx_rate_same_currency,
        test_next_field_fresh_session,
        test_next_field_skips_already_filled,
        test_next_field_payout_branch_bank,
        test_next_field_payout_branch_wallet,
        test_collection_complete,
        test_apply_update_two_level,
        test_apply_update_three_level,
        test_correction_recorded_in_audit_log,
        test_fx_recalculation_on_trigger_field,
        test_phase_resolution_all_stages,
        test_validate_amount_bounds,
        test_validate_unsupported_country,
        test_validate_email_format,
        test_validate_routing_number,
        test_validate_phone_digits,
        test_process_user_input_multi_field,
        test_process_user_input_correction,
        test_process_user_input_validation_error,
        test_receipt_blocked_when_incomplete,
        test_receipt_contains_all_sections,
        test_receipt_shows_change_log,
        test_submit_transfer_marks_submitted,
        test_get_transfer_summary_strips_none,
    ]

    passed = failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL  {test.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 48}")
    print(f"  {passed} passed  |  {failed} failed  |  {len(tests)} total")
    print(f"{'=' * 48}")
    sys.exit(1 if failed else 0)

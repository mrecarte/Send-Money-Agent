"""
Send Money Agent — Business Logic & ADK Tools

FX rate helpers, state management (apply_update, get_next_missing_field),
input validation, and the four ADK tool functions exposed to the LlmAgent.
"""
from __future__ import annotations

import contextlib
import io
import json as _json
import random
from datetime import datetime
from typing import Any

import yfinance as yf
from google.adk.tools import ToolContext

from models import (
    CorrectionEntry, SessionState, TransferState,
    COUNTRY_CURRENCY_MAP, FIELD_DISPLAY_NAMES, FEE_RATES,
    FX_TRIGGER_FIELDS, PAYOUT_SUB_MODELS,
)


# FX Helpers

def get_receive_currency(destination_country: str) -> str | None:
    """Returns the ISO 4217 receive currency for a supported destination, or None."""
    normalized = destination_country.strip().title()
    if normalized.startswith("The "):
        normalized = normalized[4:]
    return COUNTRY_CURRENCY_MAP.get(normalized)


def _fetch_yf_rate(from_ccy: str, to_ccy: str) -> float | None:
    """Fetches one direct FX rate from yfinance, silencing library noise."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rate = yf.Ticker(f"{from_ccy}{to_ccy}=X").fast_info["last_price"]
        return round(float(rate), 4)
    except Exception:
        return None


def get_fx_rate(send_currency: str, receive_currency: str) -> float | None:
    """
    Fetches a live FX rate from Yahoo Finance.

    Tries the direct pair first (e.g. INRHNL=X). If that pair is not listed,
    falls back to routing through USD (INR→USD × USD→HNL), which yfinance
    always carries for mainstream currencies. Returns None if both paths fail.
    """
    if send_currency == receive_currency:
        return 1.0

    rate = _fetch_yf_rate(send_currency, receive_currency)
    if rate is not None:
        return rate

    if send_currency != "USD" and receive_currency != "USD":
        send_to_usd = _fetch_yf_rate(send_currency, "USD")
        usd_to_recv = _fetch_yf_rate("USD", receive_currency)
        if send_to_usd is not None and usd_to_recv is not None:
            return round(send_to_usd * usd_to_recv, 4)

    return None


def recalculate_computed_fields(transfer: TransferState) -> None:
    """
    Recomputes receive_currency, fx_rate, fees, and receive_amount in place.
    Must be called whenever a field in FX_TRIGGER_FIELDS changes.
    """
    if transfer.destination_country:
        transfer.receive_currency = get_receive_currency(transfer.destination_country)

    if transfer.send_currency and transfer.receive_currency:
        transfer.fx_rate = get_fx_rate(transfer.send_currency, transfer.receive_currency)

    if transfer.send_amount and transfer.payout_method:
        transfer.fees = round(transfer.send_amount * FEE_RATES.get(transfer.payout_method, 0.020), 2)

    if transfer.send_amount and transfer.fx_rate and transfer.fees is not None:
        transfer.receive_amount = round((transfer.send_amount - transfer.fees) * transfer.fx_rate, 2)


# State Helpers

def get_next_missing_field(session_state: SessionState) -> tuple[str, str] | None:
    """
    Returns the next (dot_path, field_name) not yet collected, in collection order.
    Returns None when all required fields are filled.
    """
    t, s, r, c, p = (
        session_state.transfer,
        session_state.sender,
        session_state.recipient,
        session_state.compliance,
        session_state.payout,
    )

    if not t.destination_country: return ("transfer", "destination_country")
    if not t.send_amount:         return ("transfer", "send_amount")
    if not t.send_currency:       return ("transfer", "send_currency")
    if not t.payout_method:       return ("transfer", "payout_method")

    if t.payout_method == "bank_deposit":
        bd = p.bank_details
        if not bd or not bd.bank_name:  return ("payout.bank_details", "bank_name")
        if not bd.account_number:       return ("payout.bank_details", "account_number")
        if not bd.routing_number:       return ("payout.bank_details", "routing_number")
    elif t.payout_method == "mobile_wallet":
        wd = p.wallet_details
        if not wd or not wd.wallet_provider:  return ("payout.wallet_details", "wallet_provider")
        if not wd.wallet_phone_number:        return ("payout.wallet_details", "wallet_phone_number")
    elif t.payout_method == "cash_pickup":
        cd = p.cash_pickup_details
        if not cd or not cd.pickup_city: return ("payout.cash_pickup_details", "pickup_city")

    if not s.legal_name: return ("sender", "legal_name")
    if not s.phone:      return ("sender", "phone")
    if not s.email:      return ("sender", "email")

    if not r.legal_name:             return ("recipient", "legal_name")
    if not r.phone:                  return ("recipient", "phone")
    if not r.relationship_to_sender: return ("recipient", "relationship_to_sender")

    if not c.purpose_of_transfer: return ("compliance", "purpose_of_transfer")
    if not c.source_of_funds:     return ("compliance", "source_of_funds")

    return None


def is_collection_complete(session_state: SessionState) -> bool:
    """Returns True when every required field has been collected."""
    return get_next_missing_field(session_state) is None


def apply_update(session_state: SessionState, updates: dict[str, Any]) -> list[str]:
    """
    Applies dot-notation field updates to session state.

    Supports 2-level paths ("transfer.send_amount") and
    3-level paths ("payout.bank_details.account_number").
    Records corrections for any field overwritten with a new value.
    Triggers FX recalculation when any FX_TRIGGER_FIELDS field changes.

    Returns the list of field paths actually changed.
    """
    changed_fields: list[str] = []
    fx_recalculation_needed   = False

    for dotted_path, new_value in updates.items():
        if new_value is None:
            continue
        parts = dotted_path.split(".")
        try:
            if len(parts) == 2:
                object_name, field_name = parts
                target_object = getattr(session_state, object_name)
            elif len(parts) == 3:
                object_name, sub_object_name, field_name = parts
                parent_object = getattr(session_state, object_name)
                target_object = getattr(parent_object, sub_object_name)
                if target_object is None:
                    target_object = PAYOUT_SUB_MODELS[sub_object_name]()
                    setattr(parent_object, sub_object_name, target_object)
            else:
                continue

            current_value = getattr(target_object, field_name, None)
            if current_value != new_value:
                if current_value is not None:
                    session_state.corrections.append(CorrectionEntry(
                        field=dotted_path, old_value=current_value, new_value=new_value,
                    ))
                setattr(target_object, field_name, new_value)
                changed_fields.append(dotted_path)
                if field_name in FX_TRIGGER_FIELDS:
                    fx_recalculation_needed = True

        except (AttributeError, KeyError):
            continue

    if fx_recalculation_needed:
        recalculate_computed_fields(session_state.transfer)

    return changed_fields


def _resolve_current_phase(session_state: SessionState) -> str:
    """Maps current collection state to one of the four phase names."""
    next_field = get_next_missing_field(session_state)
    if next_field is None:                                return "confirmation"
    if next_field[0].startswith(("transfer", "payout")): return "transfer"
    if next_field[0].startswith(("sender", "recipient")): return "people"
    return "compliance"


# Validators (called by process_user_input before writing to state)

def _load_session(tool_context: ToolContext) -> SessionState:
    """Deserialises session state from ADK's state dict."""
    raw = tool_context.state.get("session", {})
    return SessionState(**raw) if raw else SessionState()


def _validate_fields(
    updates: dict[str, Any],
    session_state: SessionState | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """
    Rules-based validation for LLM-extracted field values.

    Returns (valid_updates, error_messages). Invalid fields are excluded from
    valid_updates so they are never written to state.
    """
    valid  = dict(updates)
    errors: list[str] = []

    if "transfer.send_amount" in updates:
        try:
            amt = float(updates["transfer.send_amount"])
            if amt <= 0:
                errors.append("Send amount must be greater than zero.")
                valid.pop("transfer.send_amount")
            else:
                send_ccy = (
                    str(updates.get("transfer.send_currency", "")).strip().upper()
                    or (session_state.transfer.send_currency if session_state else None)
                )
                _LIMIT_USD = 10_000
                if send_ccy and send_ccy != "USD":
                    usd_rate = get_fx_rate(send_ccy, "USD")
                    if usd_rate is not None:
                        usd_equiv = round(amt * usd_rate, 2)
                        if usd_equiv > _LIMIT_USD:
                            errors.append(
                                f"{amt:,.0f} {send_ccy} (≈ ${usd_equiv:,.2f} USD) exceeds "
                                f"the single-transfer limit of $10,000 USD."
                            )
                            valid.pop("transfer.send_amount")
                else:
                    if amt > _LIMIT_USD:
                        errors.append("Single-transfer limit is $10,000 USD. Please split into multiple transfers.")
                        valid.pop("transfer.send_amount")
        except (TypeError, ValueError):
            errors.append("Send amount must be a valid number.")
            valid.pop("transfer.send_amount")

    if "transfer.destination_country" in updates:
        country = str(updates["transfer.destination_country"]).strip().title()
        if country not in COUNTRY_CURRENCY_MAP:
            supported = "Mexico, Guatemala, Honduras, Colombia, El Salvador, Dominican Republic, Nicaragua"
            errors.append(
                f"Felix does not yet support transfers to {country}. "
                f"Supported destinations: {supported}."
            )
            valid.pop("transfer.destination_country")

    if "transfer.send_currency" in updates:
        code = str(updates["transfer.send_currency"]).strip().upper()
        if len(code) != 3 or not code.isalpha():
            errors.append(f"'{code}' is not a valid ISO 4217 currency code (e.g. USD, EUR, GBP).")
            valid.pop("transfer.send_currency")

    for name_key in ("sender.legal_name", "recipient.legal_name"):
        if name_key in updates:
            name = str(updates[name_key]).strip()
            if len(name.split()) < 2:
                errors.append(f"Please provide a full legal name (first and last name) for '{name}'.")
                valid.pop(name_key)

    if "sender.email" in updates:
        email = str(updates["sender.email"]).strip()
        parts = email.split("@")
        if len(parts) != 2 or "." not in parts[1]:
            errors.append(f"'{email}' does not appear to be a valid email address.")
            valid.pop("sender.email")

    for phone_key in ("sender.phone", "recipient.phone"):
        if phone_key in updates:
            phone  = str(updates[phone_key]).strip()
            digits = "".join(ch for ch in phone if ch.isdigit())
            if len(digits) < 7:
                errors.append(f"'{phone}' does not appear to be a valid phone number (too few digits).")
                valid.pop(phone_key)

    sender_phone    = valid.get("sender.phone")    or (session_state.sender.phone    if session_state else None)
    recipient_phone = valid.get("recipient.phone") or (session_state.recipient.phone if session_state else None)
    if sender_phone and recipient_phone and sender_phone == recipient_phone:
        errors.append("The sender and recipient cannot have the same phone number.")
        valid.pop("sender.phone",    None)
        valid.pop("recipient.phone", None)

    if "payout.bank_details.routing_number" in updates:
        raw_rtn = str(updates["payout.bank_details.routing_number"]).strip()
        rtn = "".join(ch for ch in raw_rtn if ch.isdigit())
        if len(rtn) != 9:
            errors.append(f"'{raw_rtn}' is not a valid 9-digit US routing number.")
            valid.pop("payout.bank_details.routing_number")
        else:
            valid["payout.bank_details.routing_number"] = rtn

    return valid, errors


# ADK Tools

def process_user_input(extracted_fields_json: str, tool_context: ToolContext) -> str:
    """
    Validates and applies LLM-extracted field updates to the session state.

    Returns routing context (current_phase, next_field_label) so the agent
    always knows exactly what to collect next.
    """
    session_state = _load_session(tool_context)

    try:
        updates = _json.loads(extracted_fields_json)
    except _json.JSONDecodeError:
        updates = {}

    valid_updates, validation_errors = _validate_fields(updates, session_state)

    corrections_before    = len(session_state.corrections)
    changed_fields        = apply_update(session_state, valid_updates)
    this_turn_corrections = {c.field for c in session_state.corrections[corrections_before:]}
    newly_filled          = [f for f in changed_fields if f not in this_turn_corrections]

    tool_context.state["session"] = session_state.model_dump()

    next_field       = get_next_missing_field(session_state)
    complete         = next_field is None
    next_field_path  = f"{next_field[0]}.{next_field[1]}" if next_field else None
    next_field_label = FIELD_DISPLAY_NAMES.get(next_field_path, next_field[1]) if next_field else None

    return _json.dumps({
        "newly_filled":        newly_filled,
        "corrections":         list(this_turn_corrections),
        "validation_errors":   validation_errors,
        "next_field_path":     next_field_path,
        "next_field_label":    next_field_label,
        "current_phase":       _resolve_current_phase(session_state),
        "collection_complete": complete,
    })


def format_transfer_receipt(tool_context: ToolContext) -> str:
    """
    Builds the structured receipt string from current session state.
    Generates and persists a unique confirmation number.
    Guards against being called before all required fields are collected.
    """
    session_state = _load_session(tool_context)

    if not is_collection_complete(session_state):
        return (
            "RECEIPT NOT READY — collection is still in progress. "
            "Continue gathering the required fields before generating the receipt."
        )

    conf_num = f"TXN-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
    session_state.confirmation.confirmation_number = conf_num
    tool_context.state["session"] = session_state.model_dump()

    t, s, r, p, c = (
        session_state.transfer, session_state.sender, session_state.recipient,
        session_state.payout,   session_state.compliance,
    )

    payout_detail_lines = ""
    if t.payout_method == "bank_deposit" and p.bank_details:
        bd = p.bank_details
        payout_detail_lines = (
            f"\nBank Name        : {bd.bank_name}"
            f"\nAccount Number   : {bd.account_number}"
            f"\nRouting Number   : {bd.routing_number}"
        )
    elif t.payout_method == "mobile_wallet" and p.wallet_details:
        wd = p.wallet_details
        payout_detail_lines = (
            f"\nWallet Provider  : {wd.wallet_provider}"
            f"\nWallet Phone     : {wd.wallet_phone_number}"
        )
    elif t.payout_method == "cash_pickup" and p.cash_pickup_details:
        payout_detail_lines = f"\nPickup City      : {p.cash_pickup_details.pickup_city}"

    change_log = ""
    if session_state.corrections:
        entries = "".join(
            f"\n• {FIELD_DISPLAY_NAMES.get(cor.field, cor.field)}: "
            f"{cor.old_value} → {cor.new_value}  (at {cor.timestamp})"
            for cor in session_state.corrections
        )
        change_log = "\n\nCHANGE LOG\n────────────────────────────────────────────" + entries

    return (
        "\n╔══════════════════════════════════════════╗"
        "\n                   TRANSFER CONFIRMATION            "
        "\n        ╚══════════════════════════════════════════╝"
        f"\n\nConfirmation #   : {conf_num}"
        "\n\nTRANSFER DETAILS"
        "\n────────────────────────────────────────────"
        f"\nAmount Sent      : {t.send_amount:.2f} {t.send_currency}"
        f"\nAmount Received  : {f'{t.receive_amount:.2f}' if t.receive_amount is not None else 'N/A'} {t.receive_currency or ''}"
        f"\nExchange Rate    : 1 {t.send_currency} = {t.fx_rate if t.fx_rate is not None else 'N/A'} {t.receive_currency or ''}"
        f"\nFees             : {f'{t.fees:.2f}' if t.fees is not None else 'N/A'} {t.send_currency}"
        f"\nPayout Method    : {t.payout_method.replace('_', ' ').title() if t.payout_method else 'N/A'}"
        f"{payout_detail_lines}"
        "\n\nSENDER"
        "\n────────────────────────────────────────────"
        f"\nName             : {s.legal_name}"
        f"\nPhone            : {s.phone}"
        f"\nEmail            : {s.email}"
        "\n\nRECIPIENT"
        "\n────────────────────────────────────────────"
        f"\nName             : {r.legal_name}"
        f"\nPhone            : {r.phone}"
        f"\nRelationship     : {r.relationship_to_sender}"
        "\n\nCOMPLIANCE"
        "\n────────────────────────────────────────────"
        f"\nPurpose          : {c.purpose_of_transfer}"
        f"\nSource of Funds  : {c.source_of_funds}"
        f"{change_log}"
        "\n\nSTATUS: PENDING YOUR APPROVAL"
        "\n────────────────────────────────────────────"
        "\nDo you confirm this transfer? (yes / no)"
    )


def submit_transfer(tool_context: ToolContext) -> str:
    """Marks the transfer as submitted after the user confirms consent."""
    session_state = _load_session(tool_context)
    session_state.confirmation.consent_given = True
    session_state.confirmation.submitted     = True
    tool_context.state["session"] = session_state.model_dump()
    return _json.dumps({
        "submitted":           True,
        "confirmation_number": session_state.confirmation.confirmation_number,
    })


def get_transfer_summary(tool_context: ToolContext) -> str:
    """
    Returns a compact JSON snapshot of only the fields collected so far.
    """
    session = _load_session(tool_context)
    t, s, r, p, c = (
        session.transfer, session.sender, session.recipient,
        session.payout,   session.compliance,
    )

    def _strip_none(d: dict) -> dict:
        return {k: v for k, v in d.items() if v is not None}

    payout_snapshot: dict = {}
    if p.bank_details:        payout_snapshot["bank"]        = _strip_none(p.bank_details.model_dump())
    if p.wallet_details:      payout_snapshot["wallet"]      = _strip_none(p.wallet_details.model_dump())
    if p.cash_pickup_details: payout_snapshot["cash_pickup"] = _strip_none(p.cash_pickup_details.model_dump())

    return _json.dumps({
        "transfer":         _strip_none(t.model_dump()),
        "sender":           _strip_none(s.model_dump()),
        "recipient":        _strip_none(r.model_dump()),
        "payout":           payout_snapshot,
        "compliance":       _strip_none(c.model_dump()),
        "current_phase":    _resolve_current_phase(session),
        "corrections_made": len(session.corrections),
    }, indent=2)

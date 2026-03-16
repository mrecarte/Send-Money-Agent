"""
Send Money Agent — State Models & Domain Constants

Pydantic models that represent the full session state, plus static configuration
(supported corridors, fee rates, field labels).
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# Pydantic State Models
# Single source of truth for the entire session. None = not yet collected.

class BankDetails(BaseModel):
    bank_name:      Optional[str] = None
    account_number: Optional[str] = None
    routing_number: Optional[str] = None


class WalletDetails(BaseModel):
    wallet_provider:     Optional[str] = None
    wallet_phone_number: Optional[str] = None


class CashPickupDetails(BaseModel):
    pickup_city: Optional[str] = None


class TransferState(BaseModel):
    """Fields related to the transfer and all computed financials."""
    destination_country: Optional[str]   = None
    send_amount:         Optional[float] = None
    send_currency:       Optional[str]   = None
    payout_method: Optional[Literal["bank_deposit", "cash_pickup", "mobile_wallet"]] = None
    # Computed — populated automatically, never asked from the user
    receive_currency: Optional[str]   = None
    fx_rate:          Optional[float] = None
    fees:             Optional[float] = None
    receive_amount:   Optional[float] = None


class SenderState(BaseModel):
    legal_name: Optional[str] = None
    phone:      Optional[str] = None
    email:      Optional[str] = None


class RecipientState(BaseModel):
    legal_name:             Optional[str] = None
    phone:                  Optional[str] = None
    relationship_to_sender: Optional[str] = None


class PayoutState(BaseModel):
    """Only one sub-object will be populated, depending on payout_method."""
    bank_details:        Optional[BankDetails]        = None
    wallet_details:      Optional[WalletDetails]      = None
    cash_pickup_details: Optional[CashPickupDetails]  = None


class ComplianceState(BaseModel):
    purpose_of_transfer: Optional[str] = None
    source_of_funds:     Optional[str] = None


class ConfirmationState(BaseModel):
    confirmation_number: Optional[str]  = None
    consent_given:       Optional[bool] = None
    submitted:           bool           = False


class CorrectionEntry(BaseModel):
    """Immutable record of a single mid-conversation field correction — audit trail."""
    field:     str
    old_value: Any
    new_value: Any
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


class SessionState(BaseModel):
    """Complete state of a money transfer session, persisted in ADK's session.state."""
    transfer:     TransferState     = Field(default_factory=TransferState)
    sender:       SenderState       = Field(default_factory=SenderState)
    recipient:    RecipientState    = Field(default_factory=RecipientState)
    payout:       PayoutState       = Field(default_factory=PayoutState)
    compliance:   ComplianceState   = Field(default_factory=ComplianceState)
    confirmation: ConfirmationState = Field(default_factory=ConfirmationState)
    corrections:  list[CorrectionEntry] = Field(default_factory=list)


# Domain Constants

# Felix's currently supported send corridors
COUNTRY_CURRENCY_MAP: dict[str, str] = {
    "Mexico":             "MXN",
    "Guatemala":          "GTQ",
    "Honduras":           "HNL",
    "Colombia":           "COP",
    "El Salvador":        "USD",
    "Dominican Republic": "DOP",
    "Nicaragua":          "NIO",
}

# Changing any of these fields triggers FX recalculation
FX_TRIGGER_FIELDS: frozenset[str] = frozenset({
    "destination_country",
    "send_currency",
    "send_amount",
    "payout_method",
})

# Fee rate applied to send_amount per payout method
FEE_RATES: dict[str, float] = {
    "bank_deposit":  0.015,   # 1.5% — cheapest, fully digital
    "mobile_wallet": 0.020,   # 2.0% — mid-tier
    "cash_pickup":   0.025,   # 2.5% — most expensive, physical handling
}

# Human-readable labels for every collectible field (used in agent responses)
FIELD_DISPLAY_NAMES: dict[str, str] = {
    "transfer.destination_country":               "destination country",
    "transfer.send_amount":                       "amount you want to send",
    "transfer.send_currency":                     "currency you are sending (e.g. USD)",
    "transfer.payout_method":                     "payout method — bank deposit, cash pickup, or mobile wallet",
    "payout.bank_details.bank_name":              "recipient's bank name",
    "payout.bank_details.account_number":         "recipient's bank account number",
    "payout.bank_details.routing_number":         "recipient's bank routing number",
    "payout.wallet_details.wallet_provider":      "mobile wallet provider (e.g. Tigo Money, OXXO Pay)",
    "payout.wallet_details.wallet_phone_number":  "mobile wallet phone number",
    "payout.cash_pickup_details.pickup_city":     "city where the recipient will pick up the cash",
    "sender.legal_name":                          "your full legal name",
    "sender.phone":                               "your phone number",
    "sender.email":                               "your email address",
    "recipient.legal_name":                       "recipient's full legal name",
    "recipient.phone":                            "recipient's phone number",
    "recipient.relationship_to_sender":           "your relationship to the recipient (e.g. sister, friend)",
    "compliance.purpose_of_transfer":             "purpose of this transfer (e.g. family support, business)",
    "compliance.source_of_funds":                 "source of funds (e.g. salary, savings)",
}

# Maps payout sub-object names to their model class for lazy initialisation
PAYOUT_SUB_MODELS: dict[str, type] = {
    "bank_details":        BankDetails,
    "wallet_details":      WalletDetails,
    "cash_pickup_details": CashPickupDetails,
}

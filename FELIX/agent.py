"""
Send Money Agent

Run:  python agent.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from models import SessionState
from tools import (
    _resolve_current_phase,
    format_transfer_receipt,
    get_transfer_summary,
    process_user_input,
    submit_transfer,
)

os.environ["GOOGLE_API_KEY"]            = "ENTER API KEY"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "0"

logging.getLogger("google.adk").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("google_genai.types").setLevel(logging.ERROR)  # suppress function_call part warning

MODEL    = "gemini-2.5-flash"
APP_NAME = "send_money_agent"


# Phase Instructions
# _BASE_EXTRACTION_RULES is shared across all four phases.
# PHASE_INSTRUCTIONS is injected as the system instruction before every LLM
# call via before_model_callback, giving one agent the persona of four.

_BASE_EXTRACTION_RULES = (
    "CANCEL RULE (check this first, every turn):\n"
    "If the user says 'cancel', 'stop', 'never mind', 'forget it', or 'start over':\n"
    "  Reply: 'Transfer cancelled. Let me know whenever you are ready to try again.'\n"
    "  Do NOT call any tools. Do not ask any more questions.\n\n"
    "HOW TO HANDLE EVERY OTHER USER MESSAGE:\n\n"
    "STEP 1 — EXTRACT. Pull every transfer-related field mentioned, even if out of order.\n"
    "Users may give you many pieces at once ('send $500 USD to my sister Maria in Mexico').\n"
    "Capture ALL of them. Use these dot-notation keys:\n"
    "   TRANSFER     : transfer.destination_country | transfer.send_amount (float)\n"
    "                | transfer.send_currency (3-letter ISO) | transfer.payout_method\n"
    "                  (values: bank_deposit | cash_pickup | mobile_wallet)\n"
    "   PAYOUT BANK  : payout.bank_details.bank_name | payout.bank_details.account_number\n"
    "                | payout.bank_details.routing_number\n"
    "   PAYOUT WALLET: payout.wallet_details.wallet_provider\n"
    "                | payout.wallet_details.wallet_phone_number\n"
    "   PAYOUT CASH  : payout.cash_pickup_details.pickup_city\n"
    "   SENDER       : sender.legal_name | sender.phone | sender.email\n"
    "   RECIPIENT    : recipient.legal_name | recipient.phone\n"
    "                | recipient.relationship_to_sender\n"
    "   COMPLIANCE   : compliance.purpose_of_transfer | compliance.source_of_funds\n\n"
    "STEP 2 — MANDATORY TOOL CALL: You MUST call process_user_input every single turn.\n"
    "Pass all extracted fields as a JSON object. If nothing was extractable, pass {}.\n"
    "Do NOT respond to the user before calling process_user_input. No exceptions.\n\n"
    "STEP 3 — RESPOND using ONLY the tool response. Never use your own memory of what\n"
    "was said — the tool is the single source of truth for what has been collected.\n"
    "   • Ask for next_field_label — this is the ONLY question to ask.\n"
    "   • NEVER ask again for any field listed in newly_filled or corrections.\n"
    "   • NEVER ask for destination_country if it appears in newly_filled.\n"
    "   a) validation_errors non-empty → relay the error, re-ask that specific field.\n"
    "   b) corrections non-empty → say 'Updated! [what changed].' then ask next_field_label.\n"
    "   c) newly_filled has 2+ fields → briefly confirm what was captured, ask next_field_label.\n"
    "      Example: 'Got it — $500 USD to Mexico via bank deposit. What is the bank name?'\n"
    "   d) newly_filled has 1 field → just ask next_field_label naturally.\n"
    "   e) collection_complete true → say the phase handoff line.\n\n"
    "ALWAYS: one question per turn. Never reveal JSON, field keys, or tool internals.\n"
)

PHASE_INSTRUCTIONS: dict[str, str] = {
    "transfer": (
        "You are Felix, a Transfer Specialist at a secure international remittance platform.\n"
        "YOUR SOLE FOCUS THIS PHASE: destination country, send amount, send currency,\n"
        "payout method, and the corresponding payout account details.\n"
        "When collection_complete is true for this phase, say exactly:\n"
        "  'Great, I have all the transfer details. Now I need a few details about "
        "the sender and recipient.'\n\n"
        + _BASE_EXTRACTION_RULES
    ),
    "people": (
        "You are Felix, an Identity Verification Specialist at a secure remittance platform.\n"
        "YOUR SOLE FOCUS THIS PHASE: sender legal name, phone, email — and recipient legal\n"
        "name, phone, and relationship to sender.\n"
        "These fields are required by federal anti-money-laundering regulations (BSA/FinCEN).\n"
        "When collection_complete is true for this phase, say exactly:\n"
        "  'Perfect. Just two quick compliance questions and we will be ready to confirm.'\n\n"
        + _BASE_EXTRACTION_RULES
    ),
    "compliance": (
        "You are Felix, a Compliance Officer at a secure remittance platform.\n"
        "YOUR SOLE FOCUS THIS PHASE: purpose of transfer and source of funds.\n"
        "These are standard anti-money-laundering fields required under BSA/FinCEN rules.\n"
        "Be matter-of-fact and professional — this is routine procedure, not an accusation.\n"
        "When collection_complete is true for this phase, say exactly:\n"
        "  'Thank you. Let me prepare your transfer summary for review.'\n\n"
        + _BASE_EXTRACTION_RULES
    ),
    "confirmation": (
        "You are Felix, a Transfer Confirmationist at a secure remittance platform.\n"
        "YOUR SOLE FOCUS THIS PHASE: present the receipt and collect explicit user consent.\n\n"
        "STEPS:\n"
        "1. Call format_transfer_receipt (no arguments) and present it EXACTLY as returned.\n"
        "2. If the user says yes / confirm / approve:\n"
        "   Call submit_transfer, then reply:\n"
        "   'Your transfer has been submitted! Confirmation #: [number]. "
        "You will receive an email shortly. Thank you for using Felix!'\n"
        "3. If the user says no / cancel:\n"
        "   Reply: 'Transfer cancelled. Let me know if you would like to start over.'\n"
        "4. If the user wants to correct a detail:\n"
        "   Call process_user_input with the correction, acknowledge the change,\n"
        "   then call format_transfer_receipt again and re-display the updated receipt.\n\n"
        + _BASE_EXTRACTION_RULES
    ),
}


# Callback & Agent Definition

def phase_persona_injector(callback_context, llm_request) -> None:
    """
    ADK before_model_callback — fires before every LLM call.

    Reads the current phase from session state and injects the corresponding
    phase-specific system instruction, giving one agent the focused behaviour
    of four specialised agents — using only ADK's documented callback API.
    """
    raw     = callback_context.state.get("session", {})
    session = SessionState(**raw) if raw else SessionState()
    phase   = _resolve_current_phase(session)

    if llm_request.config is None:
        llm_request.config = genai_types.GenerateContentConfig()
    llm_request.config.system_instruction = PHASE_INSTRUCTIONS[phase]
    llm_request.config.temperature        = 0.1   # deterministic tool-call behaviour
    return None


send_money_agent = LlmAgent(
    name                  = "send_money_agent",
    model                 = MODEL,
    instruction           = PHASE_INSTRUCTIONS["transfer"],  # fallback; overridden by callback
    before_model_callback = phase_persona_injector,
    tools                 = [process_user_input, format_transfer_receipt, submit_transfer, get_transfer_summary],
)


# Runner & Conversation Loop

session_service = InMemorySessionService()
runner          = Runner(agent=send_money_agent, app_name=APP_NAME, session_service=session_service)


def _is_submitted(session) -> bool:
    """Returns True once the user has confirmed and the transfer is submitted."""
    raw = session.state.get("session", {})
    return bool(raw) and SessionState(**raw).confirmation.submitted


async def _run_one_turn(user_id: str, session_id: str, user_text: str) -> str:
    """Sends one user message to the runner and returns the agent's reply."""
    message       = genai_types.Content(role="user", parts=[genai_types.Part(text=user_text)])
    response_text = ""
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=message):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = event.content.parts[0].text
    return response_text


async def run_conversation() -> None:
    user_id    = "user_001"
    session_id = f"session_{uuid.uuid4().hex[:8]}"

    await session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id, state={}
    )
    print("Agent : Hi! I am here to help you send money. Where would you like to send it today?\n")

    _EXIT_WORDS = {"exit", "quit", "bye", "goodbye"}
    while True:
        user_input = input("You   : ").strip()
        if not user_input:
            continue
        if user_input.lower() in _EXIT_WORDS:
            print("\nAgent : Goodbye!\n")
            break

        response = await _run_one_turn(user_id, session_id, user_input)
        print(f"\nAgent : {response}\n")

        if response and "Transfer cancelled" in response:
            break

        session = await session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
        if _is_submitted(session):
            break


if __name__ == "__main__":
    asyncio.run(run_conversation())

# Send Money Agent

An agent that guides a user through a complete international
money transfer. Built with a single `LlmAgent` and Gemini 2.5 Flash.

---

## How It Works

The agent collects all required fields across 4 phases:

1. **Transfer** — destination country, amount, currency, payout method + details
2. **People** — sender and recipient identity (name, phone, email)
3. **Compliance** — purpose of transfer, source of funds
4. **Confirmation** — receipt review and final consent

A `before_model_callback` swaps the system instruction before each LLM call,
giving one agent the focused behaviour of four specialised agents.

---

## Project Structure

```
models.py       — Pydantic state models and domain constants
tools.py        — FX helpers, validators, state logic, ADK tools
agent.py        — Phase instructions, LlmAgent, runner, conversation loop
test_agent.py   — 25 unit tests (no API key required)
```

---

## Requirements

```bash
pip install google-adk google-genai pydantic yfinance
```

---

## Run the Agent

```bash
python agent.py
```

Type naturally. Say `cancel` or `bye` to exit at any time.

## Run the Unit Tests

```bash
python test_agent.py
```

Expected output: `25 passed | 0 failed | 25 total`





------------------------------------------------------------
To Enter your own API key, please paste it in FELIX/agent.py on line 27. 

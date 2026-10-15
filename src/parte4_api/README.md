# Merchant Complaint & Churn Intelligence Service (`src/parte4_api/`)

Microservice for automated classification, urgency triage, and churn risk mitigation for merchants in payment platforms.

## Architecture

```text
parte4_api/
├── main.py        # FastAPI app, REST endpoints & dependency injection
├── agent.py       # Agno AI Agent + deterministic tools + guardrails
├── schemas.py     # Pydantic v2 data models (request / response validation)
└── README.md      # API documentation
```

## Quick Start

```bash
# Option A: Run with real OpenAI LLM
export OPENAI_API_KEY=your_key_here
uvicorn src.parte4_api.main:app --reload --port 8000

# Option B: Run in Deterministic Mock Mode (zero API dependencies)
export MOCK_LLM=1
uvicorn src.parte4_api.main:app --reload --port 8000
```

## Endpoints

### `GET /health`
Returns service status, model configuration, and version.

### `POST /classify`
Analyzes a merchant message, incorporates CRM merchant context, redact PII, checks for prompt injection, and outputs structured classification with urgency score (1-5) and human escalation flags.

### `POST /classify/batch`
Batch asynchronous processing for high-throughput queues.

## Enterprise Guardrails
- **Prompt Injection Defense**: Multi-pattern regex and adversarial intent detection.
- **Automated PII Redaction**: Masks credit card numbers, emails, and phone numbers before LLM processing.
- **Human Escalation Queue**: Persists high-urgency or ambiguous complaints to `outputs/human_review_queue.jsonl`.

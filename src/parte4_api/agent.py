"""
Agente Agno para clasificar reclamaciones de merchants.

Requisitos (ver `STATEMENT.md` Parte 4):
  1. `Agent` de Agno con `instructions` claras y `response_model` Pydantic.
  2. >= 2 tools custom:
       - get_merchant_context(merchant_id) -> dict
       - flag_for_human_review(merchant_id, reason) -> dict
  3. Guardrail prompt injection.
  4. PII redaction antes del LLM.

Sustituye los TODO por tu implementaciÃ³n. Docs Agno: https://docs.agno.com
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# Constantes y helpers
# -----------------------------------------------------------------------------
# Repo root = parents[2] desde agent.py (.../repo/src/parte4_api/agent.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore (?:all )?previous instructions", re.IGNORECASE),
    re.compile(r"\bsystem\s*:", re.IGNORECASE),
    re.compile(r"\bdisregard\b.*\b(prompt|instructions)\b", re.IGNORECASE),
    re.compile(r"ignora las instrucciones anteriores", re.IGNORECASE),
    re.compile(r"olvida las reglas anteriores", re.IGNORECASE),
    re.compile(r"eres un asistente (sin reglas|libre)", re.IGNORECASE),
]

# Orden importante: mÃ¡s especÃ­fico (card) â†’ menos especÃ­fico (phone) â†’ email.
# Si phone se aplicase antes que card, capturarÃ­a 16 dÃ­gitos con espacios.
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "card": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    "phone": re.compile(r"\+?\d[\d\s\-]{7,11}\d"),
    "email": re.compile(r"[\w\.-]+@[\w\.-]+\.\w+"),
    "cpf_cnpj": re.compile(r"\b(?:\d{3}\.\d{3}\.\d{3}-\d{2}|\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b")
}


def is_mock_mode() -> bool:
    """True si debe usarse el LLM stub (no llama a OpenAI)."""
    return os.environ.get("MOCK_LLM", "").lower() in {"1", "true", "yes"}


def detect_prompt_injection(text: str) -> bool:
    """Devuelve True si `text` contiene patrones tÃ­picos de prompt injection."""
    return any(p.search(text) for p in PROMPT_INJECTION_PATTERNS)


def redact_pii(text: str) -> str:
    # Reemplazamos secuencialmente iterando el dict (orden de especificidad)
    redacted = text
    for label, pattern in PII_PATTERNS.items():
        redacted = pattern.sub(f"[{label.upper()}]", redacted)
    return redacted


# -----------------------------------------------------------------------------
# Tools del agente
# -----------------------------------------------------------------------------
_MERCHANTS_CACHE: dict[int, dict[str, Any]] | None = None


def _load_merchants() -> dict[int, dict[str, Any]]:
    global _MERCHANTS_CACHE
    if _MERCHANTS_CACHE is None:
        path = DATA_DIR / "merchants_context.json"
        if not path.exists():
            _MERCHANTS_CACHE = {}
        else:
            data = json.loads(path.read_text())
            _MERCHANTS_CACHE = {m["merchant_id"]: m for m in data}
    return _MERCHANTS_CACHE


def get_merchant_context(merchant_id: int) -> dict[str, Any]:
    """
    Tool del agente: devuelve contexto del merchant.

    Si el merchant_id no existe, decide quÃ© hacer y documÃ©ntalo en DECISIONS.md.
    """
    merchants = _load_merchants()
    return merchants.get(merchant_id, {"merchant_id": merchant_id, "found": False})


def flag_for_human_review(merchant_id: int, reason: str) -> dict[str, Any]:
    """
    Tool del agente: registra el caso en outputs/human_review_queue.jsonl.

    Side-effect real, no mock. Appendea una lÃ­nea JSON por llamada.
    """
    queue_path = OUTPUTS_DIR / "human_review_queue.jsonl"
    record = {"merchant_id": merchant_id, "reason": reason}
    with queue_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return {"queued": True, "merchant_id": merchant_id}


# -----------------------------------------------------------------------------
# Agente Agno
# -----------------------------------------------------------------------------
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool
from .schemas import ClassifyResponse

@tool
def merchant_context_tool(merchant_id: int) -> dict:
    """Devuelve la informaciÃ³n de contexto para un merchant_id dado."""
    return get_merchant_context(merchant_id)

@tool
def flag_human_review_tool(merchant_id: int, reason: str) -> dict:
    """Marca un merchant para revisiÃ³n humana con un motivo. Ãšsala si el email implica acciÃ³n legal, tono agresivo o alta incertidumbre."""
    return flag_for_human_review(merchant_id, reason)


def build_agent(model_name: str = "gpt-4o-mini"):
    """Construye el agente Agno. Si MOCK_LLM=1, devuelve un stub."""
    if is_mock_mode():
        return _MockAgent()
    
    instructions = """Eres un agente clasificador de reclamaciones de comercios (merchants) para una plataforma adquirente de pagos y servicios financieros.
Tu objetivo es analizar el correo, extraer la intenciÃ³n principal y clasificarlo de forma estructurada.
REGLAS ESTRICTAS:
- 'category' DEBE ser uno de los permitidos (e.g., technical_issue, billing, onboarding, fraud, churn_threat, other).
- Si en el texto se menciona cancelaciÃ³n, baja de cuenta, "cancel", o "churn", clasifica SIEMPRE como 'churn_threat'.
- 'urgency' es un entero del 1 al 5. 5 es crÃ­tico (fraude, churn), 1 es baja prioridad (informativo).
- 'requires_human_escalation' es un booleano. True si la urgencia es 4 o 5, o si el email contiene amenazas legales.
- 'reasoning' es una explicaciÃ³n brevÃ­sima (max 300 caracteres) de por quÃ© elegiste esa clasificaciÃ³n.
- INSTRUCCIONES DE HERRAMIENTAS: Tienes acceso a herramientas. Ãšsalas si lo ves necesario. Si usas alguna herramienta de contexto, debes devolver 'merchant_context_used' = True. De lo contrario, False.
"""
    
    return Agent(
        model=OpenAIChat(id=model_name),
        tools=[merchant_context_tool, flag_human_review_tool],
        instructions=instructions,
        response_model=ClassifyResponse,
        show_tool_calls=False
    )


class _MockAgent:
    """
    Stub determinÃ­stico para tests offline. NO es la soluciÃ³n final.

    Reglas mÃ­nimas:
      - Si el texto detecta prompt injection â†’ category=other, urgency=1.
      - Si menciona 'cancelar' o 'churn' â†’ category=churn_threat, urgency=4.
      - En otro caso â†’ category=other, urgency=2.
    """

    def classify(self, *, merchant_id: int, email_text: str, locale: str = "es") -> dict[str, Any]:
        if detect_prompt_injection(email_text):
            return {
                "merchant_id": merchant_id,
                "category": "other",
                "urgency": 1,
                "requires_human_escalation": True,
                "reasoning": "prompt_injection_detected",
                "merchant_context_used": False,
            }
        ctx = get_merchant_context(merchant_id)
        if any(kw in email_text.lower() for kw in ["cancelar", "cancel", "churn"]):
            return {
                "merchant_id": merchant_id,
                "category": "churn_threat",
                "urgency": 4,
                "requires_human_escalation": True,
                "reasoning": "menciona intenciÃ³n de cancelar",
                "merchant_context_used": bool(ctx.get("found", True) is not False),
            }
        return {
            "merchant_id": merchant_id,
            "category": "other",
            "urgency": 2,
            "requires_human_escalation": False,
            "reasoning": "stub default",
            "merchant_context_used": False,
        }


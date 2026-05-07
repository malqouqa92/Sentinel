"""Heuristic complexity classifier. NO LLM call -- pure rules.

The score is the sum of contributions from a few orthogonal signals
(input length, command base, content keywords, requirement count).
Bands map to capability tiers; the recommended model is the cheapest
available model at-or-above that tier.

The knowledge base is consulted: if a `limitation` entry matches the
problem keywords, the tier is bumped to "advanced" regardless of
heuristic score -- once Qwen has demonstrably failed a problem class,
don't waste time trying again with a small model.
"""
from typing import Iterable

from pydantic import BaseModel

from core import config
from core.knowledge_base import KnowledgeBase
from core.logger import log_event
from core.model_registry import MODEL_REGISTRY


class ComplexityResult(BaseModel):
    score: float
    tier: str
    reasoning: list[str]
    recommended_model: str


# Per-command base score. Commands that bypass LLMs entirely score 0
# (orchestrator handles them with a builtin handler).
COMMAND_BASE_SCORES: dict[str, float] = {
    "ping": 0.0, "status": 0.0, "help": 0.0,
    "models": 0.0, "complexity": 0.0,
    "search": 0.2, "file": 0.1, "exec": 0.3,
    "extract": 0.4, "code": 0.5,
}

SIMPLE_KEYWORDS = (
    "simple", "basic", "hello world", "reverse a string", "trivial",
    "obvious", "small example", "minimal",
)
COMPLEX_KEYWORDS = (
    "concurrent", "thread-safe", "thread safe", "distributed",
    "optimize", "optimization", "architecture", "scalable",
    "high-performance", "race condition", "deadlock",
    "consensus", "transaction", "atomic", "lock-free",
)


def _count_requirements(text: str) -> int:
    """Cheap requirement counter: split on bullet/sentence markers and
    count substantive lines."""
    if not text:
        return 0
    # Bullet-style markers
    lines = []
    for chunk in text.replace("\r", "\n").split("\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith(("-", "*", "•")) or chunk[:2].rstrip(".)").isdigit():
            lines.append(chunk)
            continue
        # Sentences ending with period
        for sent in chunk.split(". "):
            s = sent.strip(" .")
            if len(s) > 8:
                lines.append(s)
    return len(lines)


def _kb_has_limitation(text: str) -> tuple[bool, list[int]]:
    """Search the KB for 'limitation' entries that match keywords from
    the problem text. Returns (matched, matched_ids)."""
    if not text or not text.strip():
        return False, []
    try:
        kb = KnowledgeBase()
        entries = kb.search(text, max_results=3)
    except Exception:
        return False, []
    matched_ids = [e.id for e in entries if e.category == "limitation"]
    return bool(matched_ids), matched_ids


def _tier_from_score(score: float) -> str:
    thresholds = config.COMPLEXITY_TIER_THRESHOLDS
    if score < thresholds["basic"]:
        return "basic"
    if score < thresholds["standard"]:
        return "standard"
    return "advanced"


def _recommend_model(tier: str) -> str:
    cfg = MODEL_REGISTRY.get_cheapest_capable(tier)
    if cfg:
        return cfg.name
    # Last resort: brain model (always present in the simplified roster)
    return "qwen3-brain"


def classify_complexity(
    command: str,
    args: dict,
    skill_name: str | None = None,
) -> ComplexityResult:
    cmd = command.lstrip("/")
    reasoning: list[str] = []

    # Auto-routing disabled -> deterministic default
    if not config.AUTO_ROUTING_ENABLED:
        return ComplexityResult(
            score=0.5, tier="standard",
            reasoning=["AUTO_ROUTING_ENABLED is False"],
            recommended_model="qwen3-brain",
        )

    # Command base
    base = COMMAND_BASE_SCORES.get(cmd, 0.4)
    score = base
    reasoning.append(f"command={cmd!r} -> base score {base}")

    # Built-in commands short-circuit -- no LLM, basic tier
    if cmd in {"ping", "status", "help", "models", "complexity"}:
        return ComplexityResult(
            score=0.0, tier="basic",
            reasoning=reasoning + ["builtin command, no model needed"],
            recommended_model=_recommend_model("basic"),
        )

    # Input length signal
    text = (args or {}).get("text") or (args or {}).get("problem") or ""
    text_len = len(text)
    if text_len < 100:
        score -= 0.1
        reasoning.append(f"text length {text_len} (<100) -> -0.1")
    elif text_len < 500:
        reasoning.append(f"text length {text_len} (100-500) -> 0.0")
    elif text_len < 2000:
        score += 0.1
        reasoning.append(f"text length {text_len} (500-2000) -> +0.1")
    else:
        score += 0.2
        reasoning.append(f"text length {text_len} (2000+) -> +0.2")

    # Keyword content
    text_lower = text.lower()
    simple_hits = [k for k in SIMPLE_KEYWORDS if k in text_lower]
    complex_hits = [k for k in COMPLEX_KEYWORDS if k in text_lower]
    if simple_hits:
        score -= 0.1
        reasoning.append(f"simple keywords {simple_hits} -> -0.1")
    if complex_hits:
        score += 0.2
        reasoning.append(f"complex keywords {complex_hits} -> +0.2")

    # Existing code context bumps complexity
    if (args or {}).get("context") or (args or {}).get("code_context"):
        score += 0.1
        reasoning.append("existing code context provided -> +0.1")

    # Requirement count
    reqs = _count_requirements(text)
    if reqs > 3:
        bonus = 0.05 * (reqs - 3)
        score += bonus
        reasoning.append(
            f"{reqs} requirements (>3) -> +{bonus:.2f}"
        )

    # Clamp
    score = max(0.0, min(1.0, score))
    tier = _tier_from_score(score)

    # KB limitation override
    matched, matched_ids = _kb_has_limitation(text)
    if matched and tier != "advanced":
        reasoning.append(
            f"KB limitation entries match this class "
            f"(ids={matched_ids}) -> bumping tier to advanced"
        )
        tier = "advanced"

    recommended = _recommend_model(tier)
    reasoning.append(f"final score={score:.2f} tier={tier}")
    reasoning.append(f"recommended_model={recommended}")

    log_event(
        "SEN-system", "DEBUG", "complexity",
        f"classified command={cmd} score={score:.2f} tier={tier} "
        f"-> {recommended}",
    )
    return ComplexityResult(
        score=score, tier=tier,
        reasoning=reasoning, recommended_model=recommended,
    )

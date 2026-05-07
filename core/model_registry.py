"""Model registry: tracks the inference backends and what's actually
available right now. Sentinel never crashes because a model is missing;
it just falls through to the next tier.

Models are declared in `config.MODEL_ROSTER`. `check_availability()`
asks Ollama which models are pulled and the claude CLI whether it's
installed, then flips `available=True/False` on each entry.
"""
from typing import ClassVar

from pydantic import BaseModel, Field

from core import config
from core.logger import log_event


CAPABILITY_RANK = {"basic": 0, "standard": 1, "advanced": 2}


class ModelConfig(BaseModel):
    name: str
    model_id: str
    backend: str       # "ollama" | "claude_cli"
    context_window: int
    speed_tier: str    # "fast" | "medium" | "slow"
    capability_tier: str  # "basic" | "standard" | "advanced"
    vram_mb: int | None = None
    cost_per_1k_tokens: float = 0.0
    max_output_tokens: int = 2000
    default_temperature: float = 0.1
    available: bool = True


SPEED_RANK = {"fast": 0, "medium": 1, "slow": 2}


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ModelConfig] = {}
        self._load_from_config()

    def _load_from_config(self) -> None:
        self._models.clear()
        for raw in config.MODEL_ROSTER:
            cfg = ModelConfig(**raw)
            self._models[cfg.name] = cfg

    def get(self, name: str) -> ModelConfig | None:
        return self._models.get(name)

    def list_models(self) -> list[ModelConfig]:
        return list(self._models.values())

    def get_by_tier(self, capability: str) -> list[ModelConfig]:
        return [m for m in self._models.values()
                if m.capability_tier == capability]

    def get_cheapest_capable(
        self, min_capability: str,
    ) -> ModelConfig | None:
        """Cheapest model that meets the capability threshold AND is
        currently available. Tie-breaks on speed_tier (faster first)
        then on cost (cheaper first)."""
        threshold = CAPABILITY_RANK.get(min_capability, 0)
        candidates = [
            m for m in self._models.values()
            if m.available
            and CAPABILITY_RANK.get(m.capability_tier, 0) >= threshold
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda m: (
            m.cost_per_1k_tokens,
            SPEED_RANK.get(m.speed_tier, 99),
            CAPABILITY_RANK.get(m.capability_tier, 99),
        ))
        return candidates[0]

    def check_availability(self) -> dict[str, bool]:
        """Probe each backend and update `available` on every model.
        Returns the {name -> available} map after probing."""
        ollama_models = _probe_ollama_tags()
        claude_cli_ok = _probe_claude_cli()

        out: dict[str, bool] = {}
        for cfg in self._models.values():
            if cfg.backend == "ollama":
                cfg.available = cfg.model_id in ollama_models
            elif cfg.backend == "claude_cli":
                cfg.available = claude_cli_ok
            else:
                cfg.available = False
            out[cfg.name] = cfg.available
        log_event(
            "SEN-system", "INFO", "model_registry",
            f"availability check: {out}",
        )
        return out


def _probe_ollama_tags() -> set[str]:
    """Return the set of model_ids currently pulled in Ollama. Includes
    both the raw tag (e.g. 'sentinel-brain:latest') AND the bare base
    (e.g. 'sentinel-brain') so config entries that omit ':latest' match.
    On any failure return empty set."""
    try:
        from core.llm import OllamaClient  # local import: avoid cycle
        client = OllamaClient()
        status, body = client._request("GET", "/api/tags", timeout=3)
        if status != 200:
            return set()
        names: set[str] = set()
        for m in (body or {}).get("models", []):
            full = m.get("name") or m.get("model") or ""
            if not full:
                continue
            names.add(full)
            # Add the base form too so config entries without :tag match.
            if ":" in full:
                names.add(full.split(":", 1)[0])
        return names
    except Exception as e:
        log_event(
            "SEN-system", "WARNING", "model_registry",
            f"ollama tags probe failed: {type(e).__name__}: {e}",
        )
        return set()


def _probe_claude_cli() -> bool:
    """Return True iff a `claude` binary can be located on this system."""
    try:
        from skills.code_assist import _find_claude_cli
        return _find_claude_cli() is not None
    except Exception:
        return False


# Module-level singleton.
MODEL_REGISTRY = ModelRegistry()

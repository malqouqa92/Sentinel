"""AgentRegistry: holds all configured agents (skill compositions).

Auto-discovers YAML definitions in the `agents/` directory at startup.
Each YAML file produces one AgentConfig, validated against the schema
and registered. Failures (invalid YAML, missing skills, schema errors)
are logged at ERROR level and skipped -- one bad config doesn't block
the registry.
"""
from pathlib import Path

import yaml

from core import config
from core.agents import Agent, AgentConfig
from core.logger import log_event
from core.registry import SKILL_REGISTRY, SkillRegistry


class AgentRegistry:
    def __init__(self, skill_registry: SkillRegistry | None = None) -> None:
        self._agents: dict[str, Agent] = {}
        self._skill_registry = skill_registry or SKILL_REGISTRY

    def register(self, cfg: AgentConfig) -> None:
        if cfg.name in self._agents:
            raise ValueError(f"agent already registered: {cfg.name!r}")
        # Agent.__init__ validates the pipeline against the skill registry.
        agent = Agent(cfg, self._skill_registry)
        self._agents[cfg.name] = agent
        log_event(
            "SEN-system", "INFO", "registry",
            f"Registered agent: {cfg.name} "
            f"pipeline={cfg.skill_pipeline}",
        )

    def get(self, name: str) -> Agent | None:
        return self._agents.get(name)

    def has(self, name: str) -> bool:
        return name in self._agents

    def list_agents(self) -> list[dict]:
        return [
            {
                "name": a.config.name,
                "description": a.config.description,
                "skill_pipeline": a.config.skill_pipeline,
                "model": a.config.model,
            }
            for a in self._agents.values()
        ]

    def discover(
        self, agents_dir: Path | None = None
    ) -> dict[str, int]:
        if agents_dir is None:
            agents_dir = config.PROJECT_ROOT / "agents"
        summary = {"loaded": 0, "registered": 0, "errors": 0}
        if not agents_dir.exists():
            return summary

        for path in sorted(agents_dir.glob("*.yaml")):
            if path.name.startswith("_"):
                continue
            summary["loaded"] += 1
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError(
                        f"YAML root is {type(raw).__name__}, expected dict"
                    )
                cfg = AgentConfig(**raw)
            except Exception as e:
                summary["errors"] += 1
                log_event(
                    "SEN-system", "ERROR", "registry",
                    f"failed to load agent yaml {path.name}: "
                    f"{type(e).__name__}: {e}",
                )
                continue

            try:
                if cfg.name in self._agents:
                    # Re-register silently to support hot-reload in dev.
                    del self._agents[cfg.name]
                self.register(cfg)
                summary["registered"] += 1
            except Exception as e:
                summary["errors"] += 1
                log_event(
                    "SEN-system", "ERROR", "registry",
                    f"failed to register agent {cfg.name}: "
                    f"{type(e).__name__}: {e}",
                )
        return summary

    def reset(self) -> None:
        self._agents.clear()


# Module-level singleton.
AGENT_REGISTRY = AgentRegistry()

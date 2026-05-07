"""SkillRegistry: singleton holding all auto-discovered skills.

Auto-discovery scans the `skills/` directory for *.py files, imports each,
finds BaseSkill subclasses defined in that module, instantiates them, and
registers them. Failures in one skill are logged at ERROR but do not crash
the registry — other skills still load.
"""
import importlib
import inspect
import sys
from pathlib import Path

from core import config
from core.logger import log_event
from core.skills import BaseSkill


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> None:
        if skill.name in self._skills:
            raise ValueError(
                f"skill already registered: {skill.name!r}"
            )
        self._skills[skill.name] = skill
        log_event(
            "SEN-system", "INFO", "registry",
            f"Registered skill: {skill.name} v{skill.version}",
        )

    def get(self, name: str) -> BaseSkill | None:
        return self._skills.get(name)

    def has(self, name: str) -> bool:
        return name in self._skills

    def list_skills(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "description": s.description,
                "version": s.version,
                "requires_gpu": s.requires_gpu,
            }
            for s in self._skills.values()
        ]

    def discover(self, skills_dir: Path | None = None) -> dict[str, int]:
        """Import every *.py in skills_dir and register any BaseSkill
        subclasses defined in those modules.

        Files starting with `_` or `test_` are skipped. Returns a summary
        of {"loaded", "registered", "errors"}.
        """
        if skills_dir is None:
            skills_dir = config.PROJECT_ROOT / "skills"
        summary = {"loaded": 0, "registered": 0, "errors": 0}
        if not skills_dir.exists():
            return summary

        # Make sure the directory's parent is importable so
        # `import {pkg}.{module}` works.
        parent = str(skills_dir.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        package = skills_dir.name

        for path in sorted(skills_dir.glob("*.py")):
            if path.name.startswith("_") or path.name.startswith("test_"):
                continue
            module_name = f"{package}.{path.stem}"
            try:
                # Use the cached module if already loaded — avoids
                # duplicate-class identity issues where isinstance(...)
                # fails because reload() creates a fresh class object
                # but tests/callers hold a reference to the original.
                module = (
                    sys.modules[module_name]
                    if module_name in sys.modules
                    else importlib.import_module(module_name)
                )
                summary["loaded"] += 1
            except Exception as e:
                summary["errors"] += 1
                log_event(
                    "SEN-system", "ERROR", "registry",
                    f"failed to load skill module {module_name}: "
                    f"{type(e).__name__}: {e}",
                )
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if obj is BaseSkill:
                    continue
                if not issubclass(obj, BaseSkill):
                    continue
                if inspect.isabstract(obj):
                    continue
                # Only register classes that are defined IN this module —
                # avoids double-registering imports.
                if obj.__module__ != module.__name__:
                    continue
                try:
                    instance = obj()
                    if instance.name in self._skills:
                        # Already registered (e.g., re-discovery in tests).
                        # Replace silently to allow hot-reload.
                        self._skills[instance.name] = instance
                    else:
                        self.register(instance)
                        summary["registered"] += 1
                except Exception as e:
                    summary["errors"] += 1
                    log_event(
                        "SEN-system", "ERROR", "registry",
                        f"failed to instantiate skill {obj.__name__} "
                        f"from {module_name}: {type(e).__name__}: {e}",
                    )
        return summary

    def reset(self) -> None:
        """For tests: clear all registered skills."""
        self._skills.clear()


# Module-level singleton.
SKILL_REGISTRY = SkillRegistry()

import json
import shlex

from pydantic import BaseModel

from core import config, database
from core.logger import log_event
from core.telemetry import generate_trace_id


class RouteResult(BaseModel):
    trace_id: str
    route: str
    command: str
    args: dict
    status: str
    message: str
    error_code: str | None = None
    task_id: str | None = None


def _parse_args(tokens: list[str]) -> dict:
    if not tokens:
        return {}
    if not any(t.startswith("--") for t in tokens):
        return {"text": " ".join(tokens)}

    args: dict = {}
    text_parts: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            key = tok[2:].lower()
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            if nxt is None or nxt.startswith("--"):
                args[key] = True
                i += 1
            else:
                args[key] = nxt
                i += 2
        else:
            # Free text after / between flags is collected as `text`
            # (Phase 7 enhancement -- enables `/cmd --flag value "the prompt"`).
            text_parts.append(tok)
            i += 1
    if text_parts:
        args["text"] = " ".join(text_parts)
    return args


def _log_result(result: RouteResult) -> None:
    level = "INFO" if result.status == "ok" else "WARNING"
    log_event(result.trace_id, level, "router", json.dumps(result.model_dump()))


def route(raw_input: str) -> RouteResult:
    trace_id = generate_trace_id()
    log_event(trace_id, "DEBUG", "router", f"received: {raw_input!r}")

    stripped = raw_input.strip()
    if not stripped:
        result = RouteResult(
            trace_id=trace_id,
            route="",
            command="",
            args={},
            status="error",
            message="empty input — command must be the first non-whitespace token and start with '/'",
            error_code="INVALID_POSITION",
        )
        _log_result(result)
        return result

    # Phase 18: split into command + raw rest (preserving internal
    # whitespace including newlines) so multi-line text bodies (e.g.
    # pasted /gwen STEP-N recipes) survive routing intact. Old code
    # ran `stripped.split()` then `" ".join(tokens[1:])` which
    # collapsed every whitespace run into a single space -- ate
    # newlines, broke recipe parsers downstream.
    cmd_split = stripped.split(maxsplit=1)
    if not cmd_split:
        result = RouteResult(
            trace_id=trace_id,
            route="", command="", args={},
            status="error",
            message="empty input after tokenization",
            error_code="INVALID_POSITION",
        )
        _log_result(result)
        return result
    first = cmd_split[0]
    rest = cmd_split[1] if len(cmd_split) > 1 else ""

    # Quote-aware tokenization is needed only when flags are present
    # (so `--context "multi word value"` survives intact). For text-mode
    # commands like `/exec print('hi')` we MUST keep the original
    # whitespace split -- shlex would strip the single quotes and turn
    # the code into a NameError. The Phase 7 router enhancement is
    # therefore gated on `--` appearing in the input.
    if "--" in rest:
        try:
            tokens = shlex.split(stripped, posix=True)
        except ValueError:
            tokens = stripped.split()
    else:
        tokens = [first] + rest.split() if rest else [first]
    if not first.startswith("/"):
        result = RouteResult(
            trace_id=trace_id,
            route=first,
            command="",
            args={},
            status="error",
            message="command must be the first non-whitespace token and start with '/'",
            error_code="INVALID_POSITION",
        )
        _log_result(result)
        return result

    route_lower = first.lower()
    command = route_lower.lstrip("/")

    if route_lower not in config.REGISTERED_COMMANDS:
        result = RouteResult(
            trace_id=trace_id,
            route=first,
            command=command,
            args={},
            status="error",
            message=f"unknown command: {route_lower}",
            error_code="UNKNOWN_COMMAND",
        )
        _log_result(result)
        return result

    # Phase 18: when no flags are present, args["text"] is the raw rest
    # of the input with whitespace (incl. newlines) intact. Flag-parsing
    # path still uses _parse_args (tokens already shlex-split above).
    if "--" in rest:
        args = _parse_args(tokens[1:])
    else:
        args = {"text": rest} if rest else {}

    task_id = database.add_task(trace_id, command, args)

    result = RouteResult(
        trace_id=trace_id,
        route=first,
        command=command,
        args=args,
        status="ok",
        message="routed",
        error_code=None,
        task_id=task_id,
    )
    _log_result(result)
    return result

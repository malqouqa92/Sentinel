import uuid


def generate_trace_id() -> str:
    return f"SEN-{uuid.uuid4().hex[:8]}"

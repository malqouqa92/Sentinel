import sys

def show_progress(pct: float, width: int = 40) -> None:
    pct = max(0.0, min(100.0, float(pct)))
    filled = int(width * pct / 100)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    sys.stdout.write(f"[\u2588] {pct:5.1f}%")
    sys.stdout.flush()
    if pct >= 100.0:
        sys.stdout.write("\n")
        sys.stdout.flush()
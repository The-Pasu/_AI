from typing import Iterable, List


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())

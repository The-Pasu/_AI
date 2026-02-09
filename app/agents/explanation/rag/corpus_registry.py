from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class CorpusEntry:
    source: str
    note: str
    path: str
    tags: List[str] = field(default_factory=list)


AVAILABLE_CORPORA: List[CorpusEntry] = []

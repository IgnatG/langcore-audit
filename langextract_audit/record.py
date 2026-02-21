"""Audit record data structure for the audit-logging provider."""

from __future__ import annotations

import dataclasses
import hashlib
from datetime import UTC, datetime
from typing import Any


@dataclasses.dataclass(frozen=True)
class AuditRecord:
    """Immutable record of a single inference call.

    Attributes:
        model_id: The model identifier used for the inference.
        prompt_hash: SHA-256 hex digest of the prompt text.
        response_hash: SHA-256 hex digest of the response text.
        latency_ms: Wall-clock time for the inference in
            milliseconds.
        timestamp: UTC timestamp of the inference call.
        success: Whether the inference succeeded (score > 0).
        score: The score from the ``ScoredOutput``.
        token_usage: Optional token usage dict from the response.
        batch_index: Index of the prompt within the batch.
        batch_size: Total number of prompts in the batch.
        extra: Arbitrary additional metadata.
    """

    model_id: str
    prompt_hash: str
    response_hash: str
    latency_ms: float
    timestamp: str
    success: bool
    score: float | None = None
    token_usage: dict[str, int] | None = None
    batch_index: int = 0
    batch_size: int = 1
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary suitable for serialisation.

        Returns:
            A dictionary representation of the audit record.
        """
        return dataclasses.asdict(self)

    @staticmethod
    def hash_text(text: str) -> str:
        """Compute a SHA-256 hex digest of the given text.

        Parameters:
            text: The text to hash.

        Returns:
            A hexadecimal SHA-256 digest string.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def utc_now_iso() -> str:
        """Return the current UTC time as an ISO 8601 string.

        Returns:
            An ISO-formatted UTC timestamp string.
        """
        return datetime.now(UTC).isoformat()

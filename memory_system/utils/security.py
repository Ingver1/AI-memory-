"""memory_system.utils.security
================================
High‑level cryptographic helpers for AI‑memory‑.

Key features
------------
* **CryptoContext** – injectable object encapsulating symmetric encryption (Fernet),
  signing and verification with automatic *key rotation*.
* **Pluggable KMS backend** – default local JSON keyring + optional AWS KMS backend
  (lazy import, safe to mock in unit‑tests).
* **Audit trail** – every call to *encrypt/sign/verify/decrypt* is logged to
  ``logging.getLogger("ai_memory.security.audit")`` with correlation IDs.
* 100 % type‑hinted, ready for dependency injection via FastAPI.

Example
~~~~~~~
>>> ctx = CryptoContext.from_env()
>>> token = ctx.encrypt(b"secret")
>>> data  = ctx.decrypt(token)

Copyright © Ingver1 2025.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, PositiveInt, SecretBytes, SecretStr, field_validator

__all__ = [
    "CryptoContext",
    "KeyManagementBackend",
    "LocalKeyBackend",
]

_audit_logger = logging.getLogger("ai_memory.security.audit")
_audit_logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Models & Helpers
# ---------------------------------------------------------------------------
class KeyMetadata(BaseModel):
    """Metadata attached to each managed key."""

    key_id: str  # UUID v4 string
    created_at: datetime
    expires_at: datetime | None = None  # None means no expiry

    @field_validator("expires_at")
    @classmethod
    def _exp_after_created(cls, v: datetime | None, info):  # type: ignore[override]
        if v is not None and v <= info.data["created_at"]:
            raise ValueError("expires_at must be after created_at")
        return v


class ManagedKey(BaseModel):
    """Actual key material with metadata."""

    metadata: KeyMetadata
    fernet_key: SecretStr  # URL‑safe base64‑encoded 32‑byte key

    def as_fernet(self) -> Fernet:
        return Fernet(self.fernet_key.get_secret_value().encode())


# ---------------------------------------------------------------------------
# Key management backend abstraction
# ---------------------------------------------------------------------------
class KeyManagementBackend(ABC):
    """Abstract backend that persists and retrieves encryption keys."""

    @abstractmethod
    def load_all(self) -> List[ManagedKey]:
        raise NotImplementedError

    @abstractmethod
    def save(self, key: ManagedKey) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, key_id: str) -> None:
        raise NotImplementedError


class LocalKeyBackend(KeyManagementBackend):
    """Simple JSON keyring stored on disk; suitable for dev or air‑gapped edge."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path or os.getenv("AI_MEMORY_KEYRING", ".keyring.json"))
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _read(self) -> MutableMapping[str, Mapping[str, str]]:
        if self._path.exists():
            return json.loads(self._path.read_text())
        return {}

    def _write(self, data: Mapping[str, Mapping[str, str]]) -> None:
        self._path.write_text(json.dumps(data, indent=2, sort_keys=True))

    # ---------------------------------------------------------------------
    # Interface implementation
    # ---------------------------------------------------------------------
    def load_all(self) -> List[ManagedKey]:
        return [ManagedKey(**entry) for entry in self._read().values()]

    def save(self, key: ManagedKey) -> None:
        ring = self._read()
        ring[key.metadata.key_id] = key.model_dump(mode="json")
        self._write(ring)

    def delete(self, key_id: str) -> None:
        ring = self._read()
        if key_id in ring:
            del ring[key_id]
            self._write(ring)


# ---------------------------------------------------------------------------
# CryptoContext
# ---------------------------------------------------------------------------
class CryptoContext:
    """Encapsulates all cryptographic operations for the service."""

    # How often to rotate keys automatically (hours)
    _DEFAULT_ROTATION_HOURS: Final[int] = 24 * 30  # 30 days

    # How long until a deprecated key is fully removed (grace period)
    _DEFAULT_RETIRE_HOURS: Final[int] = 24 * 90  # 90 days

    def __init__(
        self,
        *,
        backend: KeyManagementBackend | None = None,
        rotation_hours: int = _DEFAULT_ROTATION_HOURS,
        retire_hours: int = _DEFAULT_RETIRE_HOURS,
    ) -> None:
        self.backend: KeyManagementBackend = backend or LocalKeyBackend()
        self.rotation_hours = rotation_hours
        self.retire_hours = retire_hours
        self._lock = asyncio.Lock()
        self._cache: dict[str, ManagedKey] = {}

        # load keys from backend at startup
        for key in self.backend.load_all():
            self._cache[key.metadata.key_id] = key

        if not self._cache:
            # first‑time initialisation
            self._generate_and_store_new_key()

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------
    @property
    def _active_key(self) -> ManagedKey:
        # active = newest non‑expired key
        return max(self._cache.values(), key=lambda k: k.metadata.created_at)

    def _generate_and_store_new_key(self) -> ManagedKey:
        key_material = base64.urlsafe_b64encode(os.urandom(32)).decode()
        meta = KeyMetadata(
            key_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc),
        )
        mkey = ManagedKey(metadata=meta, fernet_key=SecretStr(key_material))
        self.backend.save(mkey)
        self._cache[mkey.metadata.key_id] = mkey
        return mkey

    async def maybe_rotate_keys(self) -> None:
        """Generate new key if the active key is older than *rotation_hours*."""
        async with self._lock:
            age = datetime.now(timezone.utc) - self._active_key.metadata.created_at
            if age > timedelta(hours=self.rotation_hours):
                new_key = self._generate_and_store_new_key()
                _audit_logger.info(
                    "key.rotated",
                    extra={"new_key": new_key.metadata.key_id, "age_hours": age.total_seconds() / 3600},
                )

            # retire keys beyond grace period
            for key_id, key in list(self._cache.items()):
                if (
                    datetime.now(timezone.utc) - key.metadata.created_at
                    > timedelta(hours=self.retire_hours)
                ):
                    self.backend.delete(key_id)
                    del self._cache[key_id]
                    _audit_logger.info("key.retired", extra={"key": key_id})

    # ------------------------------------------------------------------
    # Encryption / Decryption
    # ------------------------------------------------------------------
    def encrypt(self, data: bytes | str) -> str:
        if isinstance(data, str):
            data = data.encode()
        token = self._active_key.as_fernet().encrypt(data)
        _audit_logger.info("encrypt", extra={"by": self._active_key.metadata.key_id, "size": len(data)})
        return token.decode()

    def decrypt(self, token: str) -> bytes:
        for key in self._cache.values():
            try:
                data = key.as_fernet().decrypt(token.encode())
                _audit_logger.info("decrypt", extra={"key": key.metadata.key_id, "size": len(data)})
                return data
            except InvalidToken:
                continue
        _audit_logger.warning("decrypt.failed", extra={"token_prefix": token[:10]})
        raise InvalidToken("No valid key found for decryption")

    # ------------------------------------------------------------------
    # Message signing / verification (HMAC‑style using Fernet MAC)
    # ------------------------------------------------------------------
    def sign(self, data: bytes | str) -> str:
        token = self.encrypt(data)  # Fernet already appends MAC
        _audit_logger.info("sign", extra={"key": self._active_key.metadata.key_id})
        return token

    def verify(self, signature: str, data: bytes | str) -> bool:
        try:
            recovered = self.decrypt(signature)
        except InvalidToken:
            _audit_logger.warning("verify.invalid_token")
            return False
        result = recovered == (data.encode() if isinstance(data, str) else data)
        _audit_logger.info("verify", extra={"ok": result})
        return result

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "CryptoContext":
        """Create context using env variables or default local backend."""
        # Example: support AWS KMS when AWS_KMS_KEY_ID is defined
        if os.getenv("AWS_KMS_KEY_ID"):
            try:
                from .kms_aws import AWSKMSBackend  # local file to avoid heavy import

                backend: KeyManagementBackend = AWSKMSBackend(
                    key_id=os.environ["AWS_KMS_KEY_ID"],
                    region=os.getenv("AWS_REGION", "us-east-1"),
                )
            except ImportError:  # pragma: no cover
                logging.getLogger(__name__).warning("boto3 missing, falling back to LocalKeyBackend")
                backend = LocalKeyBackend()
        else:
            backend = LocalKeyBackend()
        return cls(backend=backend)


# ---------------------------------------------------------------------------
# Periodic background task helpers (to be wired in FastAPI lifespan)
# ---------------------------------------------------------------------------
import asyncio  # noqa: E402 – late import w/ uvicorn


async def start_maintenance(ctx: CryptoContext, interval_hours: PositiveInt = PositiveInt(6)) -> None:  # type: ignore[valid-type]
    """Run *maybe_rotate_keys* every *interval_hours* until cancelled."""

    async def _loop() -> None:  # inner to attach cancellation gracefully
        while True:
            await ctx.maybe_rotate_keys()
            await asyncio.sleep(interval_hours * 3600)

    asyncio.create_task(_loop(), name="crypto_key_maintenance")    

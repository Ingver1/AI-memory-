"""
Security utilities including PII filtering, encryption, and authentication.
"""

import hmac
import re
import secrets
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt


class PIIPatterns:
    EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    CREDIT_CARD = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
    SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b")
    IP_ADDRESS = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
    API_KEY = re.compile(r"\b[A-Za-z0-9]{32,}\b")
    UUID = re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE
    )
    IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b")
    CREDIT_CARD_SIMPLE = re.compile(r"\b\d{13,19}\b")
    PASSPORT = re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")
    DRIVERS_LICENSE = re.compile(r"\b[A-Z]{1,2}\d{6,8}\b")


class EnhancedPIIFilter:
    def __init__(self, custom_patterns: Optional[Dict[str, re.Pattern]] = None):
        self.patterns = {
            "email": PIIPatterns.EMAIL,
            "credit_card": PIIPatterns.CREDIT_CARD,
            "ssn": PIIPatterns.SSN,
            "phone": PIIPatterns.PHONE,
            "ip_address": PIIPatterns.IP_ADDRESS,
            "api_key": PIIPatterns.API_KEY,
            "uuid": PIIPatterns.UUID,
            "iban": PIIPatterns.IBAN,
            "passport": PIIPatterns.PASSPORT,
        }
        if custom_patterns:
            self.patterns.update(custom_patterns)
        self.stats = {k: 0 for k in self.patterns}

    def detect(self, text: str) -> Dict[str, List[str]]:
        detections: Dict[str, List[str]] = {}
        for t, p in self.patterns.items():
            m = p.findall(text)
            if m:
                detections[t] = m
                self.stats[t] += len(m)
        return detections

    def redact(self, text: str, redaction_char: str = "*") -> Tuple[str, bool, List[str]]:
        found: List[str] = []
        result = text
        for t, p in self.patterns.items():
            m = p.findall(result)
            if m:
                found.append(t)
                self.stats[t] += len(m)
                result = p.sub(f"[{t.upper()}_REDACTED]", result)
        return result, bool(found), found

    def partial_redact(self, text: str, preserve_chars: int = 2) -> Tuple[str, bool, List[str]]:
        found: List[str] = []
        result = text

        def _replace(val: str) -> str:
            if len(val) <= preserve_chars * 2:
                return "*" * len(val)
            return f"{val[:preserve_chars]}{'*' * (len(val) - preserve_chars * 2)}{val[-preserve_chars:]}"

        for t, p in self.patterns.items():
            if p.search(result):
                found.append(t)
                matches = p.findall(result)
                self.stats[t] += len(matches)
                result = p.sub(lambda m: _replace(m.group(0)), result)
        return result, bool(found), found

    def get_stats(self) -> Dict[str, int]:
        return self.stats.copy()

    def reset_stats(self) -> None:
        self.stats = {k: 0 for k in self.patterns}


class SecureTokenManager:
    def __init__(
        self, secret_key: str, algorithm: str = "HS256", issuer: str = "unified-memory-system"
    ):
        if len(secret_key) < 32:
            raise SecurityError("Secret key must be at least 32 characters for security")
        allowed_algs = ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"]
        if algorithm not in allowed_algs:
            raise SecurityError(f"Algorithm {algorithm} not allowed. Use one of: {allowed_algs}")
        if algorithm.startswith("RS") and "PRIVATE KEY" not in secret_key:
            raise SecurityError("RS algorithms require RSA private key")
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.issuer = issuer
        self.revoked_tokens: set[str] = set()
        self.max_revoked_tokens = 10000

    def generate_token(
        self,
        user_id: str,
        expires_in: int = 3600,
        scopes: Optional[List[str]] = None,
        audience: str = "api",
    ) -> str:
        if not user_id or len(user_id) > 100:
            raise SecurityError("Invalid user_id")
        if not 0 < expires_in <= 86400:
            raise SecurityError("Invalid expiration time")
        now = time.time()
        payload = {
            "iss": self.issuer,
            "aud": audience,
            "sub": user_id,
            "iat": now,
            "exp": now + expires_in,
            "nbf": now,
            "jti": secrets.token_urlsafe(16),
            "scopes": scopes or [],
            "token_type": "access",
            "version": "0.8-alpha",
        }
        try:
            return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        except Exception as e:
            raise SecurityError(f"Failed to generate token: {e}") from e

    def _is_token_revoked(self, jti: str) -> bool:
        for r in self.revoked_tokens:
            if hmac.compare_digest(jti, r):
                return True
        return False

    def verify_token(self, token: str, audience: str = "api") -> Optional[Dict[str, Any]]:
        if not token or len(token) > 2048:
            raise SecurityError("Invalid token format")
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                issuer=self.issuer,
                audience=audience,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_iat": True,
                    "require": ["exp", "iat", "nbf", "iss", "aud", "sub", "jti"],
                },
            )
            jti = payload.get("jti")
            if jti and self._is_token_revoked(jti):
                raise SecurityError("Token has been revoked")
            if time.time() - payload.get("iat", 0) > 86400:
                raise SecurityError("Token too old")
            return payload
        except jwt.ExpiredSignatureError:
            raise SecurityError("Token has expired") from None
        except jwt.InvalidIssuerError:
            raise SecurityError("Invalid token issuer") from None
        except jwt.InvalidAudienceError:
            raise SecurityError("Invalid token audience") from None
        except jwt.InvalidSignatureError:
            raise SecurityError("Invalid token signature") from None
        except jwt.InvalidTokenError as e:
            raise SecurityError(f"Invalid token: {e}") from e
        except Exception as e:
            raise SecurityError(f"Token verification failed: {e}") from e

    def revoke_token(self, token: str) -> bool:
        try:
            payload = jwt.decode(
                token, self.secret_key, algorithms=[self.algorithm], options={"verify_exp": False}
            )
            jti = payload.get("jti")
            if not jti:
                return False
            if len(self.revoked_tokens) >= self.max_revoked_tokens:
                self.revoked_tokens = set(
                    list(self.revoked_tokens)[-self.max_revoked_tokens // 2 :]
                )
            self.revoked_tokens.add(jti)
            return True
        except jwt.InvalidTokenError:
            return False

    def generate_refresh_token(self, user_id: str, expires_in: int = 604800) -> str:
        now = time.time()
        payload = {
            "iss": self.issuer,
            "aud": "refresh",
            "sub": user_id,
            "iat": now,
            "exp": now + expires_in,
            "jti": secrets.token_urlsafe(16),
            "token_type": "refresh",
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "issuer": self.issuer,
            "revoked_tokens_count": len(self.revoked_tokens),
            "max_revoked_tokens": self.max_revoked_tokens,
        }


class PasswordManager:
    @staticmethod
    def hash_password(
        password: str, salt: Optional[bytes] = None, iterations: int = 150000
    ) -> Tuple[str, bytes]:
        if len(password) < 8:
            raise SecurityError("Password must be at least 8 characters")
        if salt is None:
            salt = secrets.token_bytes(32)
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=iterations)
        return kdf.derive(password.encode()).hex(), salt

    @staticmethod
    def verify_password(password: str, hashed: str, salt: bytes, iterations: int = 150000) -> bool:
        try:
            expected_hash, _ = PasswordManager.hash_password(password, salt, iterations)
            return hmac.compare_digest(expected_hash, hashed)
        except Exception:
            return False

    @staticmethod
    def generate_secure_password(length: int = 16, include_symbols: bool = True) -> str:
        if not 8 <= length <= 128:
            raise SecurityError("Password length must be between 8 and 128 characters")
        lower = "abcdefghijklmnopqrstuvwxyz"
        upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        digits = "0123456789"
        symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?" if include_symbols else ""
        alphabet = lower + upper + digits + symbols
        pwd = [
            secrets.choice(lower),
            secrets.choice(upper),
            secrets.choice(digits),
        ]
        if include_symbols:
            pwd.append(secrets.choice(symbols))
        for _ in range(length - len(pwd)):
            pwd.append(secrets.choice(alphabet))
        secrets.SystemRandom().shuffle(pwd)
        return "".join(pwd)


class EncryptionManager:
    def __init__(self, key: Optional[bytes] = None):
        if key is None:
            key = Fernet.generate_key()
        if isinstance(key, str):
            key = key.encode()
        try:
            self.fernet = Fernet(key)
        except Exception as e:
            raise SecurityError(f"Invalid encryption key: {e}") from e
        self.key = key

    def encrypt(self, data: str | bytes) -> bytes:
        if isinstance(data, str):
            data = data.encode()
        if len(data) > 10 * 1024 * 1024:
            raise SecurityError("Data too large for encryption")
        return self.fernet.encrypt(data)

    def decrypt(self, encrypted_data: bytes) -> str:
        try:
            return self.fernet.decrypt(encrypted_data).decode()
        except Exception as e:
            raise SecurityError(f"Decryption failed: {e}") from e

    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        import os

        if not os.path.exists(file_path):
            raise SecurityError("Input file does not exist")
        if os.path.getsize(file_path) > 100 * 1024 * 1024:
            raise SecurityError("File too large for encryption")
        if output_path is None:
            output_path = f"{file_path}.encrypted"
        with open(file_path, "rb") as f:
            data = f.read()
        with open(output_path, "wb") as f:
            f.write(self.fernet.encrypt(data))
        return output_path

    def decrypt_file(self, encrypted_file_path: str, output_path: Optional[str] = None) -> str:
        import os

        if not os.path.exists(encrypted_file_path):
            raise SecurityError("Encrypted file does not exist")
        if output_path is None:
            output_path = encrypted_file_path.replace(".encrypted", "")
        with open(encrypted_file_path, "rb") as f:
            encrypted = f.read()
        with open(output_path, "wb") as f:
            f.write(self.fernet.decrypt(encrypted))
        return output_path

    @classmethod
    def generate_key(cls) -> bytes:
        return Fernet.generate_key()


class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        if requests_per_minute <= 0:
            raise ValueError("Rate limit must be positive")
        self.rpm_limit = requests_per_minute
        self.requests: Dict[str, List[float]] = {}
        self.blocked: Dict[str, float] = {}
        self._cleanup_interval = 300
        self._last_cleanup = time.time()
        self._lock = threading.RLock()

    def _cleanup(self, cutoff: float) -> None:
        remove = [u for u, t in self.requests.items() if not [x for x in t if x > cutoff]]
        for u in remove:
            self.requests.pop(u, None)
        now = time.time()
        self.blocked = {ip: ts for ip, ts in self.blocked.items() if now - ts < 3600}

    def is_allowed(self, identifier: str, weight: int = 1) -> bool:
        now = time.time()
        window_start = now - 60
        with self._lock:
            if now - self._last_cleanup > self._cleanup_interval:
                self._cleanup(window_start)
                self._last_cleanup = now
            if identifier in self.blocked and now - self.blocked[identifier] < 3600:
                return False
            self.requests.setdefault(identifier, [])
            self.requests[identifier] = [t for t in self.requests[identifier] if t > window_start]
            if len(self.requests[identifier]) + weight > self.rpm_limit:
                if len(self.requests[identifier]) > self.rpm_limit * 2:
                    self.blocked[identifier] = now
                return False
            self.requests[identifier].extend([now] * weight)
            return True

    def get_remaining_requests(self, identifier: str) -> int:
        with self._lock:
            now = time.time()
            window_start = now - 60
            current = len([t for t in self.requests.get(identifier, []) if t > window_start])
            return max(0, self.rpm_limit - current)

    def reset_user(self, identifier: str) -> bool:
        with self._lock:
            self.requests.pop(identifier, None)
            self.blocked.pop(identifier, None)
            return True

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "rpm_limit": self.rpm_limit,
                "active_users": len(self.requests),
                "blocked": len(self.blocked),
                "total_requests": sum(len(v) for v in self.requests.values()),
            }


class SecurityError(Exception):
    pass


class AuthenticationError(SecurityError):
    pass


class AuthorizationError(SecurityError):
    pass

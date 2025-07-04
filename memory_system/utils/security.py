"""Security utilities including PII filtering and encryption."""

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
    """Predefined regex patterns for personally identifiable information (PII)."""
    EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    CREDIT_CARD = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
    SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    IP_ADDRESS = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
    API_KEY = re.compile(r"\b[A-Za-z0-9]{32,}\b")
    UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
    IBAN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b")
    CREDIT_CARD_SIMPLE = re.compile(r"\b\d{13,19}\b")
    PASSPORT = re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")
    DRIVERS_LICENSE = re.compile(r"\b[A-Z]{1,2}\d{6,8}\b")

class EnhancedPIIFilter:
    """Utility for detecting and redacting PII (Personally Identifiable Information) in text."""
    def __init__(self, custom_patterns: Optional[Dict[str, re.Pattern]] = None):
        # Default patterns can be extended with custom ones
        self.patterns: Dict[str, re.Pattern] = {
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
        # Statistics for detections
        self.stats: Dict[str, int] = {k: 0 for k in self.patterns}

    def detect(self, text: str) -> Dict[str, List[str]]:
        """Detect all PII occurrences in the text. Returns a dict of pattern name to list of matches."""
        detections: Dict[str, List[str]] = {}
        for name, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                detections[name] = matches
                self.stats[name] += len(matches)
        return detections

    def redact(self, text: str) -> Tuple[str, bool, List[str]]:
        """Redact all PII in the text by replacing with placeholders. Returns (redacted_text, found_any, types_found)."""
        found_types: List[str] = []
        redacted_text = text
        for name, pattern in self.patterns.items():
            if pattern.search(redacted_text):
                found_types.append(name)
                self.stats[name] += len(pattern.findall(redacted_text))
                redacted_text = pattern.sub(f"[{name.upper()}_REDACTED]", redacted_text)
        return redacted_text, bool(found_types), found_types

    def partial_redact(self, text: str, preserve_chars: int = 2) -> Tuple[str, bool, List[str]]:
        """Partially redact PII by replacing the middle of matches with asterisks. Preserves first/last N characters."""
        found_types: List[str] = []
        redacted_text = text

        def _partial_replace(match: re.Match) -> str:
            val = match.group(0)
            if len(val) <= preserve_chars * 2:
                return "*" * len(val)
            return f"{val[:preserve_chars]}{'*' * (len(val) - preserve_chars * 2)}{val[-preserve_chars:]}"

        for name, pattern in self.patterns.items():
            if pattern.search(redacted_text):
                found_types.append(name)
                self.stats[name] += len(pattern.findall(redacted_text))
                redacted_text = pattern.sub(_partial_replace, redacted_text)
        return redacted_text, bool(found_types), found_types

# (Encryption and authentication utilities like hashing, token generation could be added here, with proper docstrings.)
# For brevity, these are not expanded upon.

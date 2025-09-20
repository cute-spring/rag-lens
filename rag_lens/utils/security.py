"""
Security utilities for RAG Lens
"""

import hashlib
import secrets
import string
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import base64
import os

from .logger import get_logger
from .errors import SecurityError

logger = get_logger(__name__)


class SecurityManager:
    """Manage security operations"""

    def __init__(self):
        self.secret_key = self._get_or_create_secret_key()
        self.session_tokens = {}

    def _get_or_create_secret_key(self) -> str:
        """Get or create secret key for encryption"""
        key_path = ".secret_key"
        if os.path.exists(key_path):
            with open(key_path, 'r') as f:
                return f.read().strip()
        else:
            key = secrets.token_urlsafe(32)
            with open(key_path, 'w') as f:
                f.write(key)
            os.chmod(key_path, 0o600)  # Read/write only for owner
            return key

    def generate_api_key(self) -> str:
        """Generate secure API key"""
        return f"ragl_{secrets.token_urlsafe(32)}"

    def validate_api_key(self, api_key: str, stored_hash: str) -> bool:
        """Validate API key against stored hash"""
        return self._hash_api_key(api_key) == stored_hash

    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256((api_key + self.secret_key).encode()).hexdigest()

    def generate_session_token(self, user_id: str, expiry_hours: int = 24) -> str:
        """Generate session token"""
        token_data = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(hours=expiry_hours)).isoformat()
        }
        token = self._encode_token(token_data)
        self.session_tokens[token] = token_data
        return token

    def validate_session_token(self, token: str) -> bool:
        """Validate session token"""
        if token not in self.session_tokens:
            return False

        token_data = self.session_tokens[token]
        expires_at = datetime.fromisoformat(token_data["expires_at"])

        if datetime.utcnow() > expires_at:
            del self.session_tokens[token]
            return False

        return True

    def _encode_token(self, data: Dict[str, Any]) -> str:
        """Encode token data"""
        json_data = str(data).encode()
        encoded = base64.urlsafe_b64encode(json_data)
        return encoded.decode()

    def sanitize_input(self, input_string: str) -> str:
        """Sanitize user input"""
        if not isinstance(input_string, str):
            raise SecurityError("Input must be a string")

        # Remove potentially dangerous characters
        sanitized = input_string.strip()
        sanitized = sanitized.replace('\x00', '')  # Remove null bytes

        # Limit length
        if len(sanitized) > 10000:  # 10KB limit
            raise SecurityError("Input too long")

        return sanitized

    def validate_file_type(self, filename: str, allowed_extensions: list) -> bool:
        """Validate file type"""
        if not filename:
            return False

        file_ext = os.path.splitext(filename)[1].lower()
        return file_ext in allowed_extensions

    def generate_csrf_token(self) -> str:
        """Generate CSRF token"""
        return secrets.token_urlsafe(32)

    def validate_csrf_token(self, token: str, session_token: str) -> bool:
        """Validate CSRF token"""
        # Simple implementation - in production, use more secure method
        return len(token) == 43 and token.isalnum()

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            from cryptography.fernet import Fernet
            f = Fernet(self._get_encryption_key())
            return f.encrypt(data.encode()).decode()
        except ImportError:
            logger.warning("Cryptography not available, using simple encoding")
            # Fallback to simple encoding (not secure for production)
            return base64.b64encode(data.encode()).decode()

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            from cryptography.fernet import Fernet
            f = Fernet(self._get_encryption_key())
            return f.decrypt(encrypted_data.encode()).decode()
        except ImportError:
            logger.warning("Cryptography not available, using simple decoding")
            # Fallback to simple decoding
            return base64.b64decode(encrypted_data.encode()).decode()

    def _get_encryption_key(self) -> bytes:
        """Get encryption key"""
        # Derive key from secret key
        return base64.urlsafe_b64encode(self.secret_key.encode()[:32].ljust(32, b'0'))

    def check_password_strength(self, password: str) -> Dict[str, Any]:
        """Check password strength"""
        if len(password) < 8:
            return {"strong": False, "reason": "Password must be at least 8 characters"}

        checks = {
            "length": len(password) >= 8,
            "uppercase": any(c.isupper() for c in password),
            "lowercase": any(c.islower() for c in password),
            "digits": any(c.isdigit() for c in password),
            "special": any(c in string.punctuation for c in password)
        }

        passed_checks = sum(checks.values())
        strong = passed_checks >= 4

        return {
            "strong": strong,
            "checks": checks,
            "score": passed_checks
        }

    def generate_secure_password(self, length: int = 16) -> str:
        """Generate secure password"""
        if length < 8:
            raise SecurityError("Password length must be at least 8 characters")

        # Use cryptographically secure random generator
        alphabet = string.ascii_letters + string.digits + string.punctuation
        password = ''.join(secrets.choice(alphabet) for _ in range(length))

        # Ensure it contains at least one of each type
        while not all([
            any(c.isupper() for c in password),
            any(c.islower() for c in password),
            any(c.isdigit() for c in password),
            any(c in string.punctuation for c in password)
        ]):
            password = ''.join(secrets.choice(alphabet) for _ in range(length))

        return password

    def sanitize_html(self, html_content: str) -> str:
        """Sanitize HTML content"""
        try:
            import bleach
            # Basic HTML sanitization
            allowed_tags = ['p', 'br', 'strong', 'em', 'u', 'h1', 'h2', 'h3', 'ul', 'ol', 'li', 'code', 'pre']
            allowed_attributes = {'a': ['href', 'title']}
            return bleach.clean(html_content, tags=allowed_tags, attributes=allowed_attributes)
        except ImportError:
            logger.warning("Bleach not available, removing HTML tags")
            # Simple tag removal
            import re
            clean = re.compile('<.*?>')
            return re.sub(clean, '', html_content)

    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    def rate_limit_check(self, identifier: str, max_requests: int = 100, window_minutes: int = 60) -> bool:
        """Simple rate limiting implementation"""
        # This is a basic implementation - in production, use Redis or similar
        current_time = datetime.utcnow()
        window_start = current_time - timedelta(minutes=window_minutes)

        # Check if identifier exists in rate limit store
        # For demo purposes, this is in-memory
        if not hasattr(self, '_rate_limits'):
            self._rate_limits = {}

        if identifier not in self._rate_limits:
            self._rate_limits[identifier] = []

        # Clean old entries
        self._rate_limits[identifier] = [
            req_time for req_time in self._rate_limits[identifier]
            if req_time > window_start
        ]

        # Check if limit exceeded
        if len(self._rate_limits[identifier]) >= max_requests:
            logger.warning(f"Rate limit exceeded for {identifier}")
            return False

        # Add current request
        self._rate_limits[identifier].append(current_time)
        return True


# Global security manager instance
security_manager = SecurityManager()
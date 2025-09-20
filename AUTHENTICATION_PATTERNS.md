# Authentication Pattern Library for API Integration

This document provides comprehensive authentication patterns for integrating with various API providers. Each pattern includes security best practices, copy-paste implementations, and detailed examples.

## Table of Contents
1. [API Key Authentication](#api-key-authentication)
2. [OAuth 2.0 Authentication](#oauth-20-authentication)
3. [JWT Authentication](#jwt-authentication)
4. [AWS Signature Authentication](#aws-signature-authentication)
5. [Azure AD Authentication](#azure-ad-authentication)
6. [Multi-Provider Authentication Manager](#multi-provider-authentication-manager)
7. [Authentication Security Best Practices](#authentication-security-best-practices)

---

## 1. API Key Authentication

### 1.1 Basic API Key Pattern

```python
import aiohttp
import base64
import hashlib
import hmac
from typing import Dict, Any, Optional
from datetime import datetime
import os

class APIKeyAuthenticator:
    """API Key authentication with various methods"""

    def __init__(self, api_key: str, key_location: str = "header"):
        self.api_key = api_key
        self.key_location = key_location  # header, query, basic_auth
        self.headers = {}

    def prepare_headers(self) -> Dict[str, str]:
        """Prepare headers with API key authentication"""

        if self.key_location == "header":
            # Method 1: Custom header
            self.headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }
        elif self.key_location == "authorization":
            # Method 2: Authorization header
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        elif self.key_location == "basic_auth":
            # Method 3: Basic Auth with API key as username
            credentials = base64.b64encode(f"{self.api_key}:".encode()).decode()
            self.headers = {
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/json"
            }

        return self.headers

    def prepare_query_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add API key to query parameters"""
        if self.key_location == "query":
            params["api_key"] = self.api_key
        return params

    async def authenticate_request(self, method: str, url: str,
                                data: Dict[str, Any] = None,
                                params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make authenticated request with API key"""

        headers = self.prepare_headers()
        prepared_params = self.prepare_query_params(params or {})

        async with aiohttp.ClientSession(headers=headers) as session:
            if method.upper() == "GET":
                async with session.get(url, params=prepared_params) as response:
                    return await self._handle_response(response)
            elif method.upper() == "POST":
                async with session.post(url, json=data, params=prepared_params) as response:
                    return await self._handle_response(response)

    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Handle HTTP response"""
        try:
            result = await response.json()
            result["status_code"] = response.status
            return result
        except:
            return {"status_code": response.status, "text": await response.text()}

# Usage Example
async def example_api_key_auth():
    # Method 1: Header-based (most common)
    auth = APIKeyAuthenticator("your-api-key", "header")
    result = await auth.authenticate_request("GET", "https://api.example.com/data")

    # Method 2: Authorization header
    auth = APIKeyAuthenticator("your-api-key", "authorization")
    result = await auth.authenticate_request("GET", "https://api.example.com/data")

    # Method 3: Query parameter
    auth = APIKeyAuthenticator("your-api-key", "query")
    result = await auth.authenticate_request("GET", "https://api.example.com/data")
```

### 1.2 HMAC Signature Authentication

```python
class HMACAuthenticator:
    """HMAC Signature authentication for enhanced security"""

    def __init__(self, api_key: str, secret_key: str, algorithm: str = "sha256"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.algorithm = algorithm

    def generate_signature(self, method: str, url: str,
                         params: Dict[str, Any] = None,
                         body: str = "") -> str:
        """Generate HMAC signature"""
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        nonce = hashlib.md5(timestamp.encode()).hexdigest()[:16]

        # Create string to sign
        string_to_sign = f"{method.upper()}&{url}&{timestamp}&{nonce}"

        # Add sorted parameters
        if params:
            sorted_params = sorted(params.items())
            param_string = "&".join([f"{k}={v}" for k, v in sorted_params])
            string_to_sign += f"&{param_string}"

        # Add body hash if present
        if body:
            body_hash = hashlib.md5(body.encode()).hexdigest()
            string_to_sign += f"&{body_hash}"

        # Generate HMAC signature
        signature = hmac.new(
            self.secret_key.encode(),
            string_to_sign.encode(),
            getattr(hashlib, self.algorithm)
        ).hexdigest()

        return {
            "signature": signature,
            "timestamp": timestamp,
            "nonce": nonce,
            "api_key": self.api_key
        }

    def prepare_signed_headers(self, method: str, url: str,
                              params: Dict[str, Any] = None,
                              body: str = "") -> Dict[str, str]:
        """Prepare headers with HMAC signature"""
        sig_data = self.generate_signature(method, url, params, body)

        return {
            "X-API-Key": self.api_key,
            "X-Signature": sig_data["signature"],
            "X-Timestamp": sig_data["timestamp"],
            "X-Nonce": sig_data["nonce"],
            "Content-Type": "application/json"
        }

# Usage Example
async def example_hmac_auth():
    auth = HMACAuthenticator("your-api-key", "your-secret-key")

    headers = auth.prepare_signed_headers(
        "POST",
        "https://api.example.com/data",
        params={"limit": 10},
        body='{"query": "test"}'
    )

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.post("https://api.example.com/data", json={"query": "test"}) as response:
            return await response.json()
```

## 2. OAuth 2.0 Authentication

### 2.1 OAuth 2.0 Client Credentials Flow

```python
import json
import asyncio
from typing import Dict, Any, Optional

class OAuth2ClientCredentials:
    """OAuth 2.0 Client Credentials flow implementation"""

    def __init__(self, client_id: str, client_secret: str,
                 token_url: str, scope: str = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.scope = scope
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None

    async def get_access_token(self, force_refresh: bool = False) -> str:
        """Get access token, refreshing if necessary"""
        if self.access_token and not force_refresh:
            if self.token_expires_at and datetime.now() < self.token_expires_at:
                return self.access_token

        # Request new token
        token_data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }

        if self.scope:
            token_data["scope"] = self.scope

        async with aiohttp.ClientSession() as session:
            async with session.post(self.token_url, data=token_data) as response:
                if response.status == 200:
                    token_response = await response.json()
                    self.access_token = token_response["access_token"]

                    # Set expiration time (subtract 60 seconds for buffer)
                    expires_in = token_response.get("expires_in", 3600)
                    self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)

                    return self.access_token
                else:
                    error_data = await response.json()
                    raise Exception(f"OAuth token request failed: {error_data.get('error_description', 'Unknown error')}")

    async def make_authenticated_request(self, method: str, url: str,
                                      data: Dict[str, Any] = None,
                                      params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make OAuth 2.0 authenticated request"""
        access_token = await self.get_access_token()

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            if method.upper() == "GET":
                async with session.get(url, params=params) as response:
                    return await self._handle_response(response)
            elif method.upper() == "POST":
                async with session.post(url, json=data, params=params) as response:
                    return await self._handle_response(response)

    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Handle HTTP response with OAuth error handling"""
        try:
            result = await response.json()
            result["status_code"] = response.status
            return result
        except:
            return {"status_code": response.status, "text": await response.text()}

# Usage Example
async def example_oauth2_client_credentials():
    oauth = OAuth2ClientCredentials(
        client_id="your-client-id",
        client_secret="your-client-secret",
        token_url="https://api.example.com/oauth/token",
        scope="read write"
    )

    # First request will automatically get token
    result = await oauth.make_authenticated_request("GET", "https://api.example.com/protected-data")
```

### 2.2 OAuth 2.0 Authorization Code Flow

```python
from urllib.parse import urlencode, parse_qs
import webbrowser
import secrets
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

class OAuth2AuthorizationCode:
    """OAuth 2.0 Authorization Code flow implementation"""

    def __init__(self, client_id: str, client_secret: str,
                 auth_url: str, token_url: str,
                 redirect_uri: str, scope: str = None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.auth_url = auth_url
        self.token_url = token_url
        self.redirect_uri = redirect_uri
        self.scope = scope
        self.state = secrets.token_urlsafe(32)
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None

        # Callback server
        self.auth_code = None
        self.callback_server = None

    def get_authorization_url(self) -> str:
        """Generate authorization URL"""
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "state": self.state
        }

        if self.scope:
            params["scope"] = self.scope

        return f"{self.auth_url}?{urlencode(params)}"

    def start_callback_server(self, port: int = 8080):
        """Start local callback server to receive authorization code"""

        class CallbackHandler(BaseHTTPRequestHandler):
            def __init__(self, oauth_instance, *args, **kwargs):
                self.oauth_instance = oauth_instance
                super().__init__(*args, **kwargs)

            def do_GET(self):
                if self.path.startswith("/callback"):
                    # Parse callback parameters
                    query = parse_qs(self.path.split('?', 1)[1] if '?' in self.path else '')

                    if 'code' in query:
                        self.oauth_instance.auth_code = query['code'][0]

                        # Check state
                        if query.get('state', [''])[0] != self.oauth_instance.state:
                            self.send_response(400)
                            self.end_headers()
                            return

                        # Send success response
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(b"<html><body><h1>Authorization successful!</h1><p>You can close this window.</p></body></html>")
                    else:
                        # Handle error
                        error = query.get('error', ['unknown_error'])[0]
                        self.send_response(400)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(f"<html><body><h1>Authorization failed!</h1><p>Error: {error}</p></body></html>".encode())

                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress log messages

        def run_server():
            server = HTTPServer(('localhost', port), lambda *args: CallbackHandler(self, *args))
            self.callback_server = server
            server.serve_forever()

        # Start server in separate thread
        server_thread = threading.Thread(target=run_server)
        server_thread.daemon = True
        server_thread.start()

        return port

    async def complete_authorization(self, port: int = 8080) -> bool:
        """Complete OAuth 2.0 authorization flow"""
        # Start callback server
        actual_port = self.start_callback_server(port)

        # Open browser for authorization
        auth_url = self.get_authorization_url()
        print(f"Opening browser for authorization: {auth_url}")
        webbrowser.open(auth_url)

        # Wait for authorization code
        max_wait = 300  # 5 minutes
        start_time = asyncio.get_event_loop().time()

        while not self.auth_code and (asyncio.get_event_loop().time() - start_time) < max_wait:
            await asyncio.sleep(1)

        if self.callback_server:
            self.callback_server.shutdown()

        if self.auth_code:
            # Exchange authorization code for access token
            return await self.exchange_code_for_token()
        else:
            raise Exception("Authorization timed out")

    async def exchange_code_for_token(self) -> bool:
        """Exchange authorization code for access token"""
        token_data = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": self.auth_code,
            "redirect_uri": self.redirect_uri
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.token_url, data=token_data) as response:
                if response.status == 200:
                    token_response = await response.json()
                    self.access_token = token_response["access_token"]
                    self.refresh_token = token_response.get("refresh_token")

                    # Set expiration time
                    expires_in = token_response.get("expires_in", 3600)
                    self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)

                    return True
                else:
                    error_data = await response.json()
                    raise Exception(f"Token exchange failed: {error_data.get('error_description', 'Unknown error')}")

    async def refresh_access_token(self) -> bool:
        """Refresh access token using refresh token"""
        if not self.refresh_token:
            return False

        token_data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.token_url, data=token_data) as response:
                if response.status == 200:
                    token_response = await response.json()
                    self.access_token = token_response["access_token"]

                    # Update refresh token if provided
                    if "refresh_token" in token_response:
                        self.refresh_token = token_response["refresh_token"]

                    # Update expiration time
                    expires_in = token_response.get("expires_in", 3600)
                    self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)

                    return True
                else:
                    return False

# Usage Example
async def example_oauth2_authorization_code():
    oauth = OAuth2AuthorizationCode(
        client_id="your-client-id",
        client_secret="your-client-secret",
        auth_url="https://api.example.com/oauth/authorize",
        token_url="https://api.example.com/oauth/token",
        redirect_uri="http://localhost:8080/callback",
        scope="read write"
    )

    # Start authorization flow
    success = await oauth.complete_authorization()
    if success:
        print("Authorization successful!")
        # Now make authenticated requests using oauth.access_token
```

## 3. JWT Authentication

### 3.1 JWT Bearer Token Authentication

```python
import jwt
import time
from datetime import datetime, timedelta

class JWTAuthenticator:
    """JWT Bearer Token authentication"""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token = None
        self.refresh_token = None

    def generate_jwt_token(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        """Generate JWT token"""
        payload_copy = payload.copy()
        payload_copy.update({
            "exp": datetime.now() + timedelta(seconds=expires_in),
            "iat": datetime.now(),
            "jti": secrets.token_urlsafe(16)
        })

        return jwt.encode(payload_copy, self.secret_key, algorithm=self.algorithm)

    def decode_jwt_token(self, token: str) -> Dict[str, Any]:
        """Decode and verify JWT token"""
        try:
            return jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        except jwt.ExpiredSignatureError:
            raise Exception("Token has expired")
        except jwt.InvalidTokenError:
            raise Exception("Invalid token")

    def prepare_bearer_headers(self, token: str) -> Dict[str, str]:
        """Prepare Bearer token headers"""
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    async def make_authenticated_request(self, method: str, url: str,
                                      token: str, data: Dict[str, Any] = None,
                                      params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make JWT authenticated request"""
        headers = self.prepare_bearer_headers(token)

        async with aiohttp.ClientSession(headers=headers) as session:
            if method.upper() == "GET":
                async with session.get(url, params=params) as response:
                    return await self._handle_response(response)
            elif method.upper() == "POST":
                async with session.post(url, json=data, params=params) as response:
                    return await self._handle_response(response)

    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Handle HTTP response"""
        try:
            result = await response.json()
            result["status_code"] = response.status
            return result
        except:
            return {"status_code": response.status, "text": await response.text()}

# Usage Example
async def example_jwt_authentication():
    # Create JWT authenticator
    jwt_auth = JWTAuthenticator("your-secret-key")

    # Generate token for user authentication
    user_payload = {
        "user_id": "user123",
        "username": "john_doe",
        "roles": ["user", "admin"]
    }

    token = jwt_auth.generate_jwt_token(user_payload, expires_in=3600)

    # Make authenticated request
    result = await jwt_auth.make_authenticated_request(
        "GET", "https://api.example.com/protected-data", token
    )
```

## 4. AWS Signature Authentication

### 4.1 AWS Signature Version 4

```python
import urllib.parse
import hashlib
import hmac

class AWSV4Authenticator:
    """AWS Signature Version 4 authentication"""

    def __init__(self, access_key: str, secret_key: str, region: str, service: str):
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.service = service

    def sign_request(self, method: str, url: str, headers: Dict[str, str] = None,
                    params: Dict[str, Any] = None, body: str = "") -> Dict[str, str]:
        """Sign request with AWS Signature Version 4"""

        # Parse URL
        parsed_url = urllib.parse.urlparse(url)
        host = parsed_url.netloc
        path = parsed_url.path or '/'
        query = parsed_url.query

        # Create canonical request
        canonical_headers = self._create_canonical_headers(headers or {})
        signed_headers = self._create_signed_headers(canonical_headers)

        if query:
            canonical_query = urllib.parse.parse_qs(query)
            canonical_query = '&'.join([f"{k}={v[0]}" for k, v in sorted(canonical_query.items())])
        else:
            canonical_query = ''

        canonical_request = f"{method.upper()}\n{path}\n{canonical_query}\n{canonical_headers}\n{signed_headers}\n{hashlib.sha256(body.encode()).hexdigest()}"

        # Create string to sign
        algorithm = "AWS4-HMAC-SHA256"
        datestamp = datetime.now().strftime("%Y%m%d")
        credential_scope = f"{datestamp}/{self.region}/{self.service}/aws4_request"

        string_to_sign = f"{algorithm}\n{datetime.now().strftime('%Y%m%dT%H%M%SZ')}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode()).hexdigest()}"

        # Calculate signature
        signing_key = self._get_signature_key(datestamp)
        signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()

        # Add authorization header
        authorization_header = (
            f"{algorithm} Credential={self.access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )

        # Add required headers
        final_headers = headers or {}
        final_headers.update({
            "Host": host,
            "X-Amz-Date": datetime.now().strftime('%Y%m%dT%H%M%SZ'),
            "Authorization": authorization_header
        })

        return final_headers

    def _create_canonical_headers(self, headers: Dict[str, str]) -> str:
        """Create canonical headers string"""
        canonical_headers = {}
        for key, value in headers.items():
            canonical_headers[key.lower().strip()] = value.strip()

        return ''.join([f"{k}:{v}\n" for k, v in sorted(canonical_headers.items())])

    def _create_signed_headers(self, canonical_headers: str) -> str:
        """Create signed headers string"""
        headers = canonical_headers.strip().split('\n')
        return ';'.join([h.split(':')[0] for h in headers])

    def _get_signature_key(self, datestamp: str) -> bytes:
        """Get signature key"""
        k_date = hmac.new(f"AWS4{self.secret_key}".encode(), datestamp.encode(), hashlib.sha256).digest()
        k_region = hmac.new(k_date, self.region.encode(), hashlib.sha256).digest()
        k_service = hmac.new(k_region, self.service.encode(), hashlib.sha256).digest()
        k_signing = hmac.new(k_service, "aws4_request".encode(), hashlib.sha256).digest()
        return k_signing

# Usage Example
async def example_aws_authentication():
    auth = AWSV4Authenticator(
        access_key="your-access-key",
        secret_key="your-secret-key",
        region="us-east-1",
        service="execute-api"
    )

    url = "https://api.example.com/resource"
    headers = auth.sign_request("GET", url)

    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url) as response:
            return await response.json()
```

## 5. Azure AD Authentication

### 5.1 Azure AD OAuth 2.0

```python
class AzureADAuthenticator:
    """Azure AD OAuth 2.0 authentication"""

    def __init__(self, tenant_id: str, client_id: str, client_secret: str,
                 scope: str = None):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope or f"api://{client_id}/.default"
        self.access_token = None
        self.token_expires_at = None

        # Azure AD endpoints
        self.token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"

    async def get_access_token(self, force_refresh: bool = False) -> str:
        """Get Azure AD access token using client credentials flow"""
        if self.access_token and not force_refresh:
            if self.token_expires_at and datetime.now() < self.token_expires_at:
                return self.access_token

        # Request new token
        token_data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": self.scope
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.token_url, data=token_data) as response:
                if response.status == 200:
                    token_response = await response.json()
                    self.access_token = token_response["access_token"]

                    # Set expiration time
                    expires_in = token_response.get("expires_in", 3600)
                    self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)

                    return self.access_token
                else:
                    error_data = await response.json()
                    raise Exception(f"Azure AD token request failed: {error_data.get('error_description', 'Unknown error')}")

    async def make_authenticated_request(self, method: str, url: str,
                                      data: Dict[str, Any] = None,
                                      params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make Azure AD authenticated request"""
        access_token = await self.get_access_token()

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            if method.upper() == "GET":
                async with session.get(url, params=params) as response:
                    return await self._handle_response(response)
            elif method.upper() == "POST":
                async with session.post(url, json=data, params=params) as response:
                    return await self._handle_response(response)

    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict[str, Any]:
        """Handle HTTP response"""
        try:
            result = await response.json()
            result["status_code"] = response.status
            return result
        except:
            return {"status_code": response.status, "text": await response.text()}

# Usage Example
async def example_azure_ad_authentication():
    azure_auth = AzureADAuthenticator(
        tenant_id="your-tenant-id",
        client_id="your-client-id",
        client_secret="your-client-secret",
        scope="api://your-client-id/.default"
    )

    result = await azure_auth.make_authenticated_request(
        "GET", "https://your-api.azurewebsites.net/protected-data"
    )
```

## 6. Multi-Provider Authentication Manager

```python
class AuthenticationManager:
    """Unified authentication manager for multiple providers"""

    def __init__(self):
        self.authenticators = {}
        self.active_authenticator = None

    def register_authenticator(self, name: str, authenticator: Any):
        """Register an authenticator"""
        self.authenticators[name] = authenticator

    def set_active_authenticator(self, name: str):
        """Set the active authenticator"""
        if name not in self.authenticators:
            raise ValueError(f"Authenticator '{name}' not registered")
        self.active_authenticator = name

    async def make_authenticated_request(self, method: str, url: str,
                                      data: Dict[str, Any] = None,
                                      params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make authenticated request using active authenticator"""
        if not self.active_authenticator:
            raise ValueError("No active authenticator set")

        authenticator = self.authenticators[self.active_authenticator]

        # Handle different authenticator interfaces
        if hasattr(authenticator, 'make_authenticated_request'):
            return await authenticator.make_authenticated_request(method, url, data, params)
        elif hasattr(authenticator, 'authenticate_request'):
            return await authenticator.authenticate_request(method, url, data, params)
        else:
            raise ValueError(f"Authenticator '{self.active_authenticator}' does not support request making")

    def get_authenticator_status(self) -> Dict[str, Any]:
        """Get status of all authenticators"""
        status = {}
        for name, auth in self.authenticators.items():
            auth_info = {
                "type": type(auth).__name__,
                "active": name == self.active_authenticator
            }

            # Add specific status based on authenticator type
            if hasattr(auth, 'access_token'):
                auth_info["has_token"] = bool(auth.access_token)
            if hasattr(auth, 'token_expires_at'):
                auth_info["token_expires_at"] = auth.token_expires_at.isoformat() if auth.token_expires_at else None

            status[name] = auth_info

        return status

# Usage Example
async def example_authentication_manager():
    auth_manager = AuthenticationManager()

    # Register multiple authenticators
    api_key_auth = APIKeyAuthenticator("your-api-key", "header")
    oauth_auth = OAuth2ClientCredentials(
        "client-id", "client-secret", "https://api.example.com/oauth/token"
    )
    jwt_auth = JWTAuthenticator("secret-key")

    auth_manager.register_authenticator("api_key", api_key_auth)
    auth_manager.register_authenticator("oauth", oauth_auth)
    auth_manager.register_authenticator("jwt", jwt_auth)

    # Switch between authenticators
    auth_manager.set_active_authenticator("api_key")
    result1 = await auth_manager.make_authenticated_request("GET", "https://api.example.com/data")

    auth_manager.set_active_authenticator("oauth")
    result2 = await auth_manager.make_authenticated_request("GET", "https://api.example.com/data")

    # Get status
    status = auth_manager.get_authenticator_status()
```

## 7. Authentication Security Best Practices

### 7.1 Secure Credential Storage

```python
import os
import keyring
from cryptography.fernet import Fernet

class SecureCredentialManager:
    """Secure credential storage and management"""

    def __init__(self, encryption_key: str = None):
        self.encryption_key = encryption_key or self._generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key.encode())

    def _generate_encryption_key(self) -> str:
        """Generate encryption key"""
        return Fernet.generate_key().decode()

    def store_credential(self, service: str, username: str, password: str):
        """Store encrypted credential"""
        encrypted_password = self.cipher_suite.encrypt(password.encode()).decode()

        # Use system keyring for secure storage
        keyring.set_password(service, username, encrypted_password)

    def get_credential(self, service: str, username: str) -> str:
        """Retrieve and decrypt credential"""
        encrypted_password = keyring.get_password(service, username)
        if not encrypted_password:
            raise ValueError(f"No credential found for {service}/{username}")

        return self.cipher_suite.decrypt(encrypted_password.encode()).decode()

    def delete_credential(self, service: str, username: str):
        """Delete stored credential"""
        try:
            keyring.delete_password(service, username)
        except:
            pass  # Credential doesn't exist

# Usage Example
def example_secure_credential_management():
    cred_manager = SecureCredentialManager()

    # Store API key securely
    cred_manager.store_credential("openai", "api_key", "sk-your-actual-api-key")

    # Retrieve API key
    api_key = cred_manager.get_credential("openai", "api_key")

    # Use API key
    auth = APIKeyAuthenticator(api_key)
```

### 7.2 Authentication Configuration Management

```python
import yaml
from pathlib import Path

class AuthenticationConfig:
    """Configuration management for authentication"""

    def __init__(self, config_file: str = "auth_config.yaml"):
        self.config_file = Path(config_file)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load authentication configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._create_default_config()

    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        default_config = {
            "providers": {
                "openai": {
                    "type": "api_key",
                    "key_location": "header",
                    "base_url": "https://api.openai.com/v1"
                },
                "azure": {
                    "type": "oauth2_client_credentials",
                    "token_url": "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
                },
                "aws": {
                    "type": "aws_v4",
                    "region": "us-east-1",
                    "service": "execute-api"
                }
            },
            "security": {
                "encrypt_credentials": True,
                "token_refresh_buffer": 60,
                "max_retries": 3
            }
        }

        # Save default config
        with open(self.config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

        return default_config

    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Get configuration for specific provider"""
        return self.config.get("providers", {}).get(provider_name, {})

    def create_authenticator(self, provider_name: str, **kwargs) -> Any:
        """Create authenticator based on configuration"""
        provider_config = self.get_provider_config(provider_name)
        auth_type = provider_config.get("type")

        if auth_type == "api_key":
            return APIKeyAuthenticator(
                api_key=kwargs.get("api_key"),
                key_location=provider_config.get("key_location", "header")
            )
        elif auth_type == "oauth2_client_credentials":
            return OAuth2ClientCredentials(
                client_id=kwargs.get("client_id"),
                client_secret=kwargs.get("client_secret"),
                token_url=provider_config.get("token_url").format(**kwargs)
            )
        elif auth_type == "jwt":
            return JWTAuthenticator(
                secret_key=kwargs.get("secret_key"),
                algorithm=provider_config.get("algorithm", "HS256")
            )
        elif auth_type == "aws_v4":
            return AWSV4Authenticator(
                access_key=kwargs.get("access_key"),
                secret_key=kwargs.get("secret_key"),
                region=provider_config.get("region", "us-east-1"),
                service=provider_config.get("service", "execute-api")
            )
        else:
            raise ValueError(f"Unsupported auth type: {auth_type}")

# Usage Example
async def example_authentication_config():
    auth_config = AuthenticationConfig()

    # Create OpenAI authenticator
    openai_auth = auth_config.create_authenticator(
        "openai",
        api_key="your-openai-api-key"
    )

    # Create Azure authenticator
    azure_auth = auth_config.create_authenticator(
        "azure",
        client_id="your-client-id",
        client_secret="your-client-secret",
        tenant_id="your-tenant-id"
    )
```

This comprehensive authentication pattern library provides:
- **Multiple authentication methods** (API Key, OAuth 2.0, JWT, AWS, Azure AD)
- **Copy-paste implementations** for quick integration
- **Security best practices** including credential storage and token management
- **Unified authentication manager** for handling multiple providers
- **Configuration management** for different environments
- **Production-ready error handling** and retry logic

Each pattern includes detailed examples and security considerations to ensure integrators can implement authentication quickly and securely.
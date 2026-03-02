# TLS Design Document

## Overview

This document explains the TLS/HTTPS implementation for the Voice Deepfake Detector API, including what TLS protects, what it doesn't protect, and how to configure it.

## What TLS Covers

### ✅ Transport Encryption
- Encrypts all HTTP traffic between client and server
- Protects API requests/responses from eavesdropping
- Prevents man-in-the-middle attacks on the transport layer

### ✅ Data Protection in Transit
- Uploaded audio files (WAV, MP3, etc.)
- Generated/cloned audio returned as base64
- API responses with predictions and confidence scores
- Authentication tokens (if implemented)

### ✅ Integrity Verification
- Ensures data is not tampered with in transit
- Certificate validation prevents impersonation

## What TLS Does NOT Cover

### ❌ Voice Authenticity
TLS encrypts the transport but does NOT verify that:
- The uploaded voice is authentic (not a deepfake)
- The generated voice clone is authorized
- The speaker has given consent

**This is the job of the ML detection models, not TLS.**

### ❌ Application-Level Security
- Input validation (file size, format, duration)
- Rate limiting (separate from TLS)
- Authentication and authorization
- Consent verification for voice cloning

### ❌ Endpoint Security
- Server infrastructure security
- Model file protection
- Database access (if used)
- Logging and audit trails

## Implementation

### Development (Self-Signed Certificate)

```bash
# Run HTTPS server with self-signed certificate
./scripts/dev_https.sh

# With auto-reload
./scripts/dev_https.sh --reload

# Custom port
./scripts/dev_https.sh --port 8443
```

The development script:
1. Generates a 4096-bit RSA key pair
2. Creates a self-signed certificate valid for 365 days
3. Starts uvicorn with SSL context
4. Browsers will show a security warning (expected for self-signed certs)

### Production (Proper Certificate)

For production, obtain certificates from:
- **Let's Encrypt** (free, automated)
- **Your organization's CA**
- **Cloud provider** (AWS ACM, Cloudflare, etc.)

```bash
# With Let's Encrypt (using certbot)
certbot certonly --standalone -d api.yourdomain.com

# Run with production certificates
uvicorn backend.app:app \
    --host 0.0.0.0 \
    --port 443 \
    --ssl-keyfile /etc/letsencrypt/live/api.yourdomain.com/privkey.pem \
    --ssl-certfile /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem
```

### Reverse Proxy (Recommended for Production)

```nginx
# nginx configuration
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

## Security Considerations

### Certificate Pinning (Optional)
For IoT devices or mobile apps, consider certificate pinning to prevent MITM attacks even with compromised CAs.

### HSTS (HTTP Strict Transport Security)
Add HSTS headers to force HTTPS:
```python
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["api.yourdomain.com"]
)

@app.middleware("http")
async def add_hsts_header(request, call_next):
    response = await call_next(request)
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
```

### TLS Configuration Best Practices

| Setting | Recommendation | Reason |
|---------|---------------|--------|
| Protocol | TLS 1.2+ | TLS 1.0/1.1 are deprecated |
| Cipher suites | ECDHE + AES-GCM | Forward secrecy + AEAD |
| Key size | 2048-bit RSA minimum | Security vs. performance |
| Certificate validity | < 397 days | Industry standard |
| HSTS | Enabled | Force HTTPS |

## Testing

### Verify TLS Configuration
```bash
# Check certificate
curl -vI https://localhost:8443/health 2>&1 | grep -E "(SSL|TLS|certificate)"

# Test SSL Labs rating (production)
# https://www.ssllabs.com/ssltest/

# Test with OpenSSL
openssl s_client -connect localhost:8443 </dev/null 2>/dev/null | openssl x509 -text
```

### API Testing with HTTPS
```bash
# Using curl with self-signed certificate
curl -k https://localhost:8443/health

# Using Python requests
import requests
requests.get("https://localhost:8443/health", verify=False)

# In production (with proper certificate)
curl https://api.yourdomain.com/health
```

## Compliance

### GDPR Considerations
- TLS is required for GDPR compliance (data protection in transit)
- Combine with encrypted storage for full protection

### Research/Academic Use
- TLS satisfies most institutional security requirements
- Document TLS usage in ethics applications

## Summary

| Layer | Protection |
|-------|-----------|
| Transport | TLS encrypts data in transit |
| Application | Rate limiting, input validation |
| ML | Deepfake detection, consent verification |
| Storage | Encrypted model files, temp file cleanup |

**Remember:** TLS is necessary but not sufficient for a secure system. Combine with application-level security measures.

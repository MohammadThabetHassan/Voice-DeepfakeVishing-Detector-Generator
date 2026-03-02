#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/dev_https.sh — Run backend with HTTPS (self-signed certificate)
#
# This script generates a self-signed SSL certificate and runs the FastAPI
# backend with HTTPS enabled for local development.
#
# ⚠️  WARNING: Self-signed certificates are NOT for production use!
#    For production, use proper certificates from Let's Encrypt or your CA.
#
# Usage:
#   ./scripts/dev_https.sh              # Run with HTTPS on port 8443
#   ./scripts/dev_https.sh --reload     # Run with auto-reload
#   ./scripts/dev_https.sh --port 8000  # Use different port
#
# The script will:
#   1. Create certs/ directory if not exists
#   2. Generate self-signed certificate (valid 365 days)
#   3. Start uvicorn with SSL context
#
# TLS Coverage:
#   ✓ Encrypts transport between browser and backend
#   ✓ Protects API keys, tokens, and uploaded audio in transit
#   ✗ Does NOT authenticate voice authenticity (deepfake can be served over HTTPS)
#   ✗ Does NOT replace consent requirements for voice cloning
# ──────────────────────────────────────────────────────────────────────────────

set -e

# Default configuration
HOST="0.0.0.0"
PORT="8443"
RELOAD=""
CERT_DIR="certs"
KEY_FILE="$CERT_DIR/server.key"
CERT_FILE="$CERT_DIR/server.crt"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --reload)
            RELOAD="--reload"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --reload        Enable auto-reload on code changes"
            echo "  --port PORT     Port to listen on (default: 8443)"
            echo "  --host HOST     Host to bind to (default: 0.0.0.0)"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  UVICORN_RELOAD  Set to 'true' to enable reload"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Create certificates directory
echo "🔐  Setting up HTTPS development environment..."
mkdir -p "$CERT_DIR"

# Generate self-signed certificate if not exists
if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ]; then
    echo "📜  Generating self-signed certificate..."
    openssl req -x509 -newkey rsa:4096 \
        -keyout "$KEY_FILE" \
        -out "$CERT_FILE" \
        -days 365 \
        -nodes \
        -subj "/C=US/ST=State/L=City/O=VoiceDeepfakeDetector/CN=localhost" \
        -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"
    echo "✅  Certificate generated:"
    echo "   - Private key: $KEY_FILE"
    echo "   - Certificate: $CERT_FILE"
    echo ""
    echo "⚠️   Note: Browsers will warn about the self-signed certificate."
    echo "    This is expected for local development."
else
    echo "✅  Using existing certificate"
fi

echo ""
echo "🚀  Starting HTTPS server on https://$HOST:$PORT"
echo "    API docs available at: https://$HOST:$PORT/docs"
echo ""

# Run uvicorn with SSL
cd "$(dirname "$0")/.."
exec python -m uvicorn backend.app:app \
    --host "$HOST" \
    --port "$PORT" \
    --ssl-keyfile "$KEY_FILE" \
    --ssl-certfile "$CERT_FILE" \
    $RELOAD

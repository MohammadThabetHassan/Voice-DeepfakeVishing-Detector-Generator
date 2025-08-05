#!/usr/bin/env python3
"""
Deepfake Vishing Detection & Response demo server.
Uses python-multipart's parse_form_data helper and returns JSON on *all* responses.
"""

import json
import os
import sys
import uuid
import traceback
from http.server import HTTPServer, SimpleHTTPRequestHandler
from io import BytesIO
import cgi
from urllib.parse import urlparse

# Ensure we can import your pipeline module
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from pipeline import detect_deepfake, generate_deepfake

UPLOAD_DIR    = os.path.join(CURRENT_DIR, 'uploads')
GENERATED_DIR = os.path.join(CURRENT_DIR, 'web_app', 'generated')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

class DeepfakeHandler(SimpleHTTPRequestHandler):
    def _set_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _json_response(self, code:int, payload:dict):
        body = json.dumps(payload).encode('utf-8')
        print(f"[DEBUG] Responding with: {body.decode('utf-8')}")  # Log response body
        self.send_response(code)
        self._set_cors_headers()
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_POST(self):
        ct = self.headers.get('Content-Type', '')
        if not ct.startswith('multipart/form-data'):
            return self._json_response(400, {"error":"Unsupported content type"})

        # Use cgi.FieldStorage to parse multipart form data
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                'REQUEST_METHOD': 'POST',
                'CONTENT_TYPE': self.headers['Content-Type'],
            }
        )

        # Extract fields and files
        fields = {}
        files = {}
        for key in form.keys():
            item = form[key]
            if item.filename:
                files[key] = item
            else:
                fields[key] = item.value

        # DEBUG: print to console
        print("FIELDS:", fields)
        print("FILES:", list(files.keys()))

        if self.path == '/detect':
            if 'audio' not in files:
                return self._json_response(400, {"error":"Missing audio file"})
            audio = files['audio']
            name  = os.path.basename(audio.filename)
            uid   = uuid.uuid4().hex
            in_p  = os.path.join(UPLOAD_DIR, f"{uid}_{name}")
            with open(in_p,'wb') as f: f.write(audio.file.read())

            try:
                pred = detect_deepfake(in_p)
                if hasattr(pred,'item'): pred = pred.item()
                # Always return a user-friendly message
                if str(pred) == '0':
                    result_msg = "Actual (real) voice detected."
                elif str(pred) == '1':
                    result_msg = "Deepfake voice detected!"
                else:
                    result_msg = f"Prediction: {pred}"
                return self._json_response(200, {"prediction": pred, "message": result_msg})
            except Exception as e:
                traceback.print_exc()
                return self._json_response(500, {"error":str(e)})

        elif self.path == '/generate':
            if 'audio' not in files or 'text' not in fields:
                return self._json_response(400, {"error":"Missing audio file or text"})
            audio = files['audio']
            text  = fields['text']
            name = os.path.basename(audio.filename)
            uid  = uuid.uuid4().hex
            in_p = os.path.join(UPLOAD_DIR, f"{uid}_{name}")
            with open(in_p,'wb') as f: f.write(audio.file.read())

            out_name = f"deepfake_{uid}_{name}"
            out_p    = os.path.join(GENERATED_DIR, out_name)
            try:
                generate_deepfake(in_p, out_p, text)
                if not os.path.exists(out_p):
                    raise RuntimeError("Generated file missing")
                return self._json_response(200, {"generated_file":f"/generated/{out_name}"})
            except Exception as e:
                traceback.print_exc()
                return self._json_response(500, {"error":str(e)})

        else:
            return self._json_response(404, {"error":"Unknown endpoint"})

    def do_GET(self):
        # Serve generated audio files
        if self.path.startswith('/generated/'):
            fp = os.path.join(GENERATED_DIR, os.path.basename(self.path))
            if os.path.exists(fp):
                self.send_response(200)
                self._set_cors_headers()
                self.send_header('Content-Type','audio/wav')
                self.end_headers()
                with open(fp,'rb') as f: self.wfile.write(f.read())
            else:
                self._json_response(404, {"error":"File not found"})
            return
        # Serve static files (css, js, etc.)
        if self.path != '/' and os.path.exists(os.path.join(CURRENT_DIR, self.path.lstrip('/'))):
            return super().do_GET()
        # For all other GET requests (including / and any query string), serve index.html
        self.send_response(200)
        self._set_cors_headers()
        self.send_header('Content-Type', 'text/html')
        with open(os.path.join(CURRENT_DIR, 'index.html'), 'rb') as f:
            content = f.read()
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def run_server(port:int=8000):
    print(f"Starting server on port {port}…")
    HTTPServer(('0.0.0.0',port), DeepfakeHandler).serve_forever()

if __name__=='__main__':
    run_server()

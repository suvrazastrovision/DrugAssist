"""Read-only HTTP clients. Raw bodies are content-addressed before parsing."""

import hashlib
import os
import time
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

from src.utils.config import ROOT, utcnow, write_json


class APIError(RuntimeError):
    """A source API request or contract failure with sanitized request details."""

    pass


class APIClient:
    """A retrying label API client that preserves raw bodies and public metadata."""

    def __init__(self, raw_dir=None, timeout=60, retries=4, session=None, sleep=time.sleep):
        """Configure raw-response storage, request limits, and injectable HTTP helpers."""
        load_dotenv(ROOT / ".env")
        self.raw_dir = Path(raw_dir or ROOT / "data/raw/http")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.timeout, self.retries, self.sleep = timeout, retries, sleep
        self.session = session or requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 FDA-Retrieval-Research/0.1", "Accept": "application/json"}
        )

    def get(self, url, params=None, provider="morvs"):
        """Fetch JSON with retries and preserve source bytes without recording API keys.

        Return the decoded response and public provenance metadata. Save the raw
        response before checking its status or parsing JSON. Raise APIError after
        exhausted retries, an unsuccessful status, or an invalid JSON response.
        """
        public_params = dict(params or {})
        request_params = dict(public_params)
        if provider == "fda" and os.getenv("FDA_API_KEY"):
            request_params["api_key"] = os.environ["FDA_API_KEY"]
        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(url, params=request_params, timeout=(10, self.timeout))
            except requests.RequestException:
                if attempt == self.retries:
                    raise APIError(f"{provider}: network request failed after retries") from None
                self.sleep(min(2**attempt, 30))
                continue
            body_hash = hashlib.sha256(response.content).hexdigest()
            raw_file = self.raw_dir / f"{body_hash}.body"
            if not raw_file.exists():
                raw_file.write_bytes(response.content)
            meta = {
                "url": url,
                "params": public_params,
                "status": response.status_code,
                "retrieved_at": utcnow(),
                "body_sha256": body_hash,
                "raw_file": str(raw_file.relative_to(ROOT))
                if raw_file.is_relative_to(ROOT)
                else str(raw_file),
                "provider": provider,
                "response_headers": {
                    k: v
                    for k, v in response.headers.items()
                    if k.lower() in {"content-type", "last-modified", "etag", "retry-after"}
                },
            }
            # Request metadata is separate from the unchanged source body.
            write_json(self.raw_dir / f"{body_hash}.{time.time_ns()}.meta.json", meta)
            if (response.status_code == 429 or response.status_code >= 500) and attempt < self.retries:
                retry_after = response.headers.get("Retry-After", "")
                self.sleep(min(float(retry_after) if retry_after.isdigit() else 2**attempt, 60))
                continue
            if not response.ok:
                raise APIError(f"{provider}: HTTP {response.status_code} (raw response preserved)")
            try:
                return response.json(), meta
            except ValueError:
                raise APIError(f"{provider}: expected JSON (raw response preserved)") from None
        raise APIError("Retry budget exhausted")

    def discover(self):
        """Fetch and validate the third-party label API's advertised contract."""
        spec, meta = self.get("https://api.morvs.ai/openapi.json")
        if not isinstance(spec, dict) or not isinstance(spec.get("paths"), dict):
            raise APIError("MORVS contract changed: invalid OpenAPI response")
        paths = {p: v for p, v in spec.get("paths", {}).items() if "/fda/labels/" in p}
        if "/api/v1/fda/labels/recent" not in paths:
            raise APIError("MORVS contract changed: recent label path missing")
        return {
            "openapi_version": spec.get("openapi"),
            "paths": paths,
            "security": spec.get("security", []),
            "provenance": meta,
            "pagination": "not documented",
            "rate_limits": "not documented; honor 429",
            "scope": "bounded recent feed and bounded name lookups; not a full FDA corpus",
        }

    def recent(self, limit=200):
        """Fetch a bounded recent-label feed and its response provenance."""
        if not 1 <= limit <= 200:
            raise ValueError("MORVS recent limit must be 1..200; pagination is not documented")
        return self.get("https://api.morvs.ai/api/v1/fda/labels/recent", {"limit": limit})

    def lookup(self, name):
        """Fetch third-party label matches for a URL-escaped drug name."""
        return self.get("https://api.morvs.ai/api/v1/fda/labels/" + quote(name, safe=""))

    def fda_label(self, set_id):
        """Fetch one authoritative label by set identifier with response provenance."""
        # A set_id lookup may return a newer version; callers must compare effective dates.
        self.sleep(0.3)
        return self.get(
            "https://api.fda.gov/drug/label.json",
            {"search": f'set_id:"{set_id}"', "limit": 1},
            provider="fda",
        )

    def fda_name(self, name, limit=10):
        """Direct authoritative fallback when the third-party lookup lacks coverage."""
        if not 1 <= limit <= 1000:
            raise ValueError("FDA limit must be 1..1000")
        escaped = name.replace("\\", "\\\\").replace('"', '\\"')
        self.sleep(0.3)
        return self.get(
            "https://api.fda.gov/drug/label.json",
            {
                "search": f'openfda.generic_name:"{escaped}"+openfda.substance_name:"{escaped}"',
                "limit": limit,
            },
            provider="fda",
        )

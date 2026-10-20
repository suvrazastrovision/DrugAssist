"""Argon2 passwords, hashed opaque tokens and a local login attempt limit."""

import hashlib
import os
import secrets
import sqlite3
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

from dotenv import load_dotenv
from pwdlib import PasswordHash

from src.auth.database import connect, initialize
from src.utils.config import ROOT


class AuthError(Exception):
    def __init__(self, status: int, detail: str, retry_after: int | None = None):
        self.status, self.detail, self.retry_after = status, detail, retry_after
        super().__init__(detail)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def public_user(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "email": row["email"],
        "created_at": datetime.fromtimestamp(row["created_at"], UTC),
    }


class AuthService:
    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], float] = time.time,
        session_seconds: int = 3600,
        attempt_limit: int = 10,
        attempt_window: int = 600,
    ):
        self.path, self.clock = Path(path), clock
        self.session_seconds = session_seconds
        self.attempt_limit, self.attempt_window = attempt_limit, attempt_window
        self.attempts: dict[str, deque] = {}
        self.lock = RLock()
        self.passwords = PasswordHash.recommended()
        # Unknown accounts still perform a password verification.
        self.dummy_hash = self.passwords.hash(secrets.token_urlsafe(32))
        initialize(self.path)

    def limit_attempts(self, client: str) -> None:
        now = self.clock()
        with self.lock:
            for key in list(self.attempts):
                bucket = self.attempts[key]
                while bucket and bucket[0] <= now - self.attempt_window:
                    bucket.popleft()
                if not bucket:
                    del self.attempts[key]
            bucket = self.attempts.setdefault(client, deque())
            if len(bucket) >= self.attempt_limit:
                retry = max(1, int(bucket[0] + self.attempt_window - now) + 1)
                raise AuthError(429, "Too many attempts. Please try again later.", retry)
            bucket.append(now)

    def signup(self, email: str, password: str) -> dict:
        email = email.strip().casefold()
        hashed = self.passwords.hash(password)
        try:
            with connect(self.path) as db:
                cursor = db.execute(
                    "INSERT INTO users(email, password_hash, created_at) VALUES (?, ?, ?)",
                    (email, hashed, int(self.clock())),
                )
                row = db.execute("SELECT * FROM users WHERE id=?", (cursor.lastrowid,)).fetchone()
        except sqlite3.IntegrityError:
            raise AuthError(409, "An account with this email already exists.") from None
        return public_user(row)

    def login(self, email: str, password: str) -> dict:
        with connect(self.path) as db:
            user = db.execute("SELECT * FROM users WHERE email=?", (email.strip().casefold(),)).fetchone()
        valid = self.passwords.verify(password, user["password_hash"] if user else self.dummy_hash)
        if not valid or user is None:
            raise AuthError(401, "Incorrect email or password.")
        token = secrets.token_urlsafe(32)
        now = int(self.clock())
        expires = now + self.session_seconds
        with connect(self.path) as db:
            db.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            db.execute(
                "INSERT INTO sessions(user_id, token_hash, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (user["id"], token_hash(token), now, expires),
            )
        return {
            "session_token": token,
            "token_type": "bearer",
            "expires_at": datetime.fromtimestamp(expires, UTC),
        }

    def current_user(self, token: str) -> dict:
        if not token or len(token) > 512:
            raise AuthError(401, "Please log in again.")
        with connect(self.path) as db:
            row = db.execute(
                "SELECT users.id, users.email, users.created_at FROM sessions "
                "JOIN users ON users.id=sessions.user_id WHERE token_hash=? AND expires_at > ?",
                (token_hash(token), int(self.clock())),
            ).fetchone()
        if row is None:
            raise AuthError(401, "Please log in again.")
        return public_user(row)

    def logout(self, token: str) -> None:
        with connect(self.path) as db:
            db.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash(token),))

    def cleanup_expired(self) -> int:
        with connect(self.path) as db:
            return db.execute("DELETE FROM sessions WHERE expires_at <= ?", (int(self.clock()),)).rowcount


def build_auth_service() -> AuthService:
    load_dotenv(ROOT / ".env")
    path = Path(os.getenv("DRUGASSIST_AUTH_DB", "data/auth/users.sqlite3"))
    return AuthService(path if path.is_absolute() else ROOT / path)

"""Check generated project artifacts against configured secret values without logging them."""

from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
secrets = [
    v for k, v in dotenv_values(ROOT / ".env").items() if ("KEY" in k or "TOKEN" in k) and v and len(v) > 12
]
paths = []
paths.extend(
    ROOT / name
    for name in ["README.md", "pyproject.toml", "uv.lock", ".env.example"]
)
for folder in ["src", "frontend", "scripts", "tests", "config", "notebooks", "reports", "results", "docs"]:
    paths.extend(
        p
        for p in (ROOT / folder).rglob("*")
        if p.is_file() and p.suffix in {".py", ".json", ".ipynb", ".html", ".csv", ".md"}
    )
for path in paths:
    text = path.read_text(encoding="utf-8")
    if any(secret in text for secret in secrets):
        raise RuntimeError("Configured credential detected in artifact: " + str(path.relative_to(ROOT)))
print("Credential scan passed for", len(paths), "source and generated artifacts")

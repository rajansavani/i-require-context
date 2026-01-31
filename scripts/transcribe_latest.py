from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ensure repo root is on the import path so "import src" works
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.transcribe_session import transcribe_latest_session  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    log = logging.getLogger("irc.script.transcribe_latest")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        log.error("OPENAI_API_KEY is not set. Add it to your .env and restart your shell.")
        return 1

    # optional overrides via env vars
    model = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe").strip() or "gpt-4o-mini-transcribe"
    language = os.getenv("OPENAI_STT_LANGUAGE", "en").strip() or "en"
    overwrite = os.getenv("OVERWRITE_TRANSCRIPTS", "0").strip() == "1"

    # run from repo root so outputs/ resolves correctly
    os.chdir(REPO_ROOT)

    log.info("Repo root: %s", REPO_ROOT.as_posix())
    log.info("Model: %s | Language: %s | Overwrite: %s", model, language, overwrite)

    session_dir, out_path = transcribe_latest_session(
        api_key=api_key,
        model=model,
        language=language,
        overwrite=overwrite,
        logger=logging.getLogger("irc.pipeline.transcribe"),
    )

    log.info("Latest session: %s", session_dir.as_posix())
    log.info("Wrote transcripts: %s", out_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

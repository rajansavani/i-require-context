from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ensure repo root is on the import path so "import src" works
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.transcribe_session import (  # noqa: E402
    find_latest_session_dir,
    transcribe_session_dir,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="transcribe the latest (or a specific) discord recording session",
    )

    parser.add_argument(
        "--session-dir",
        type=str,
        default="",
        help="path to a specific session directory under outputs/audio/sessions/<session_id>",
    )
    parser.add_argument(
        "--sessions-root",
        type=str,
        default="outputs/audio/sessions",
        help="root folder that contains session directories",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"),
        help="openai stt model name",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=os.getenv("OPENAI_STT_LANGUAGE", "en"),
        help="language hint for transcription (ex: en)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=(os.getenv("OVERWRITE_TRANSCRIPTS", "0").strip() == "1"),
        help="overwrite transcripts.json if it already exists",
    )
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        default=False,
        help="disable timestamps/segments if you want faster+smaller output",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=os.getenv("OPENAI_STT_PROMPT", "").strip(),
        help="optional stt prompt (ex: names/terms to bias transcription)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("OPENAI_STT_TEMPERATURE", "0.0")),
        help="stt temperature",
    )

    return parser.parse_args()


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

    args = _parse_args()

    # run from repo root so outputs/ resolves correctly
    os.chdir(REPO_ROOT)

    sessions_root = Path(args.sessions_root)
    session_dir: Path

    if args.session_dir.strip():
        session_dir = Path(args.session_dir)
        if not session_dir.is_absolute():
            session_dir = (REPO_ROOT / session_dir).resolve()
    else:
        found = find_latest_session_dir(sessions_root=sessions_root)
        if found is None:
            log.error("No session directories found under %s", sessions_root.as_posix())
            return 1
        session_dir = found.resolve()

    # quick sanity info for testing chunk-mode
    chunk_mode = (session_dir / "chunks.json").exists()
    log.info("Repo root: %s", REPO_ROOT.as_posix())
    log.info("Session dir: %s", session_dir.as_posix())
    log.info("Chunk mode: %s", chunk_mode)
    log.info(
        "Model: %s | Language: %s | Overwrite: %s | Timestamps: %s",
        args.model,
        args.language,
        args.overwrite,
        (not args.no_timestamps),
    )

    out_path = transcribe_session_dir(
        session_dir,
        api_key=api_key,
        model=args.model.strip() or "gpt-4o-mini-transcribe",
        language=args.language.strip() or "en",
        prompt=args.prompt if args.prompt else None,
        temperature=float(args.temperature),
        want_timestamps=(not args.no_timestamps),
        overwrite=bool(args.overwrite),
        logger=logging.getLogger("irc.pipeline.transcribe"),
    )

    log.info("Wrote transcripts: %s", out_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

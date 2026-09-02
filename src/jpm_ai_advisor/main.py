"""CLI entrypoint: run one advisory session end to end and save the transcript."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from .config import build_llm_client
from .orchestrator import run_session

PROFILES_DIR = Path(__file__).parent / "profiles"
TRANSCRIPTS_DIR = Path(__file__).parent.parent.parent / "transcripts"


def available_profiles() -> dict[str, Path]:
    return {p.stem.removeprefix("client_profile_"): p for p in PROFILES_DIR.glob("client_profile_*.json")}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    profiles = available_profiles()
    parser = argparse.ArgumentParser(description="Run a simulated financial advisory session.")
    parser.add_argument("--profile", choices=sorted(profiles), default="conservative")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-turns", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile_path = available_profiles()[args.profile]
    profile = json.loads(profile_path.read_text())

    llm = build_llm_client(provider=args.provider, model=args.model)
    result = run_session(llm=llm, profile=profile, max_client_turns=args.max_turns)

    print(result.to_markdown())

    TRANSCRIPTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = TRANSCRIPTS_DIR / f"{timestamp}_{args.profile}.md"
    out_path.write_text(result.to_markdown())
    print(f"\nSaved transcript to {out_path}", file=sys.stderr)

    return 0 if result.resolved else 1


if __name__ == "__main__":
    raise SystemExit(main())

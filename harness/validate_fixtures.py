#!/usr/bin/env python3
"""Validate audio fixtures through the SERVER'S OWN loading path.

Runs sglang_omni.preprocessing.audio.ensure_audio_list_async (the exact
function the Qwen3-Omni preprocessor calls on the top-level `audios` field,
target_sr=16000) on each fixture. Passing here means the H100 server will
accept and decode these files.

Usage: validate_fixtures.py --fixtures harness/fixtures
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "sglang-omni"))

from sglang_omni.preprocessing.audio import ensure_audio_list_async  # noqa: E402


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", required=True)
    a = p.parse_args()
    fdir = Path(a.fixtures)
    manifest = json.loads((fdir / "fixtures.json").read_text())
    for tag, meta in manifest["fixtures"].items():
        path = fdir / meta["file"]
        loaded = await ensure_audio_list_async([str(path)], target_sr=16000)
        assert len(loaded) == 1, f"{tag}: expected 1 audio, got {len(loaded)}"
        arr = loaded[0]
        dur = len(arr) / 16000
        drift = abs(dur - meta["duration_s"])
        assert drift < 0.1, f"{tag}: duration drift {drift:.2f}s"
        print(f"{tag}: server-path load OK — {dur:.2f}s @16kHz "
              f"(manifest {meta['duration_s']}s, {len(meta['utterances'])} utts)")
    print("all fixtures pass the server loading path")


if __name__ == "__main__":
    asyncio.run(main())

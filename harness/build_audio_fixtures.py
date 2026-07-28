#!/usr/bin/env python3
"""Build the three audio->text fixtures from LibriSpeech test-clean.

Contract row (#1018, verbatim): "at least three PCM fixtures spanning short,
medium, and long input". Output: 16 kHz mono PCM16 WAV + fixtures.json
manifest (source corpus, utterance IDs, durations, sha256) for the run
manifest and the results write-up.

Deterministic: fixed speaker/chapter, utterances concatenated in ID order
until each duration target is met. Rebuildable on any box from the same tar.

Usage: build_audio_fixtures.py --tar data/librispeech/test-clean.tar.gz \
           --out harness/fixtures
   or: build_audio_fixtures.py --parquet <clean/test/0000.parquet> --out harness/fixtures
       (HF openslr/librispeech_asr, same underlying FLACs, same utterance IDs)
"""

import argparse
import hashlib
import io
import json
import tarfile
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

SPEAKER, CHAPTER = "1089", "134686"
TARGETS = {"short": 5.0, "medium": 30.0, "long": 180.0}
SR = 16000


def utts_from_tar(tar_path: str) -> list[tuple[str, np.ndarray]]:
    prefix = f"LibriSpeech/test-clean/{SPEAKER}/{CHAPTER}/"
    utts: list[tuple[str, np.ndarray]] = []
    with tarfile.open(tar_path) as tf:
        names = sorted(n for n in tf.getnames()
                       if n.startswith(prefix) and n.endswith(".flac"))
        # long target needs ~180s; one chapter may not be enough — extend
        # with the speaker's other chapters in ID order.
        extra = sorted(n for n in tf.getnames()
                       if n.startswith(f"LibriSpeech/test-clean/{SPEAKER}/")
                       and n.endswith(".flac") and not n.startswith(prefix))
        for name in names + extra:
            data, sr = sf.read(io.BytesIO(tf.extractfile(name).read()),
                               dtype="int16")
            assert sr == SR, f"{name}: unexpected sample rate {sr}"
            assert data.ndim == 1, f"{name}: not mono"
            utts.append((name.split("/")[-1].removesuffix(".flac"), data))
            if sum(len(d) for _, d in utts) / SR > TARGETS["long"] + 30:
                break
    return utts


def utts_from_parquet(pq_path: str) -> list[tuple[str, np.ndarray]]:
    import pyarrow.parquet as pq
    table = pq.read_table(pq_path, columns=["id", "audio"])
    rows = [(i.as_py(), au.as_py()) for i, au in
            zip(table["id"], table["audio"])
            if i.as_py().startswith(f"{SPEAKER}-")]
    rows.sort(key=lambda r: r[0])
    # same ordering rule as the tar path: target chapter first, then the rest
    rows.sort(key=lambda r: 0 if r[0].startswith(f"{SPEAKER}-{CHAPTER}-") else 1)
    utts: list[tuple[str, np.ndarray]] = []
    for utt_id, audio in rows:
        data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="int16")
        assert sr == SR and data.ndim == 1, utt_id
        utts.append((utt_id, data))
        if sum(len(d) for _, d in utts) / SR > TARGETS["long"] + 30:
            break
    return utts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tar", default=None)
    p.add_argument("--parquet", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    assert bool(a.tar) != bool(a.parquet), "exactly one of --tar/--parquet"
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = a.tar or a.parquet
    src_sha = hashlib.sha256(Path(src).read_bytes()).hexdigest()
    utts = utts_from_tar(a.tar) if a.tar else utts_from_parquet(a.parquet)

    manifest = {
        "source": ("LibriSpeech test-clean (https://www.openslr.org/12)"
                   if a.tar else
                   "LibriSpeech test-clean via HF openslr/librispeech_asr "
                   "@ 71cacbfb7e2354c4226d01e70d77d5fca3d04ba1, clean/test/0000.parquet"),
        "archive": Path(src).name,
        "archive_sha256": src_sha,
        "speaker": SPEAKER,
        "sample_rate": SR,
        "format": "16 kHz mono PCM16 WAV",
        "fixtures": {},
    }
    for tag, target_s in TARGETS.items():
        chosen, total = [], 0
        for utt_id, data in utts:
            chosen.append((utt_id, data))
            total += len(data)
            if total / SR >= target_s:
                break
        audio = np.concatenate([d for _, d in chosen])
        path = out_dir / f"fixture_{tag}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(audio.tobytes())
        manifest["fixtures"][tag] = {
            "file": path.name,
            "duration_s": round(len(audio) / SR, 2),
            "utterances": [u for u, _ in chosen],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(f"{tag}: {manifest['fixtures'][tag]['duration_s']}s "
              f"({len(chosen)} utterances) -> {path}")

    (out_dir / "fixtures.json").write_text(json.dumps(manifest, indent=1))
    print(f"manifest -> {out_dir / 'fixtures.json'}")


if __name__ == "__main__":
    main()

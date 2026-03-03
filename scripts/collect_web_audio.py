#!/usr/bin/env python3
"""
Collect labeled audio from web sources for model training.

Capabilities:
1. YouTube search/download via yt-dlp (query based)
2. Podcast RSS parsing + episode audio download
3. Audio normalization to WAV mono 16k via ffmpeg
4. Labeled metadata export for reproducible training

Output layout:
  training/web_data/
    raw/{real,fake}/...
    processed/{real,fake}/...
    metadata.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = {
    "real": {
        "youtube_queries": [
            "human podcast interview english",
            "phone call conversation real voice",
        ],
        "youtube_urls": [],
        "podcast_feeds": [
            "https://feeds.npr.org/510289/podcast.xml",
            "https://feeds.simplecast.com/54nAGcIl",
        ],
    },
    "fake": {
        "youtube_queries": [
            "AI voice clone demo",
            "text to speech voice demo",
        ],
        "youtube_urls": [],
        "podcast_feeds": [],
    },
}


def _slug(value: str, max_len: int = 48) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    if not value:
        value = "item"
    return value[:max_len]


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def _ensure_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _safe_ext_from_url(url: str) -> str:
    lower = url.lower()
    for ext in (".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".webm", ".mp4"):
        if ext in lower:
            return ext
    return ".bin"


def _normalize_audio(src: Path, dst: Path, max_duration_s: int) -> tuple[bool, str]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-t",
        str(max_duration_s),
        str(dst),
    ]
    try:
        _run(cmd, check=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or "").strip()[:500]
        return False, err or "ffmpeg conversion failed"


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return DEFAULT_CONFIG
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _parse_podcast_feed(feed_url: str, limit: int) -> list[dict[str, str]]:
    req = urllib.request.Request(
        feed_url,
        headers={"User-Agent": "VoiceDeepfakeDatasetCollector/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    root = ET.fromstring(raw)

    items = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        audio_url = None
        for elem in item.iter():
            tag = elem.tag.lower()
            if tag.endswith("enclosure"):
                candidate = elem.attrib.get("url", "").strip()
                if candidate:
                    audio_url = candidate
                    break
        if audio_url:
            items.append({"title": title or "episode", "url": audio_url})
        if len(items) >= limit:
            break
    return items


def _download_url(url: str, dst: Path) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "VoiceDeepfakeDatasetCollector/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp, dst.open("wb") as f:
            shutil.copyfileobj(resp, f)
        return True, ""
    except Exception as e:
        return False, str(e)[:500]


def _collect_youtube_query(
    query: str,
    label: str,
    raw_dir: Path,
    max_downloads: int,
    min_duration_s: int,
    dry_run: bool,
) -> list[dict[str, str]]:
    rows = []
    source_id = _sha1(f"yt-query:{label}:{query}")
    if dry_run:
        rows.append(
            {
                "label": label,
                "source_type": "youtube_query",
                "source_ref": query,
                "url": f"ytsearch:{query}",
                "raw_path": "",
                "processed_path": "",
                "status": "dry_run",
                "error": "",
            }
        )
        return rows

    out_tpl = str(raw_dir / f"{label}_yt_{source_id}_%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--ignore-errors",
        "--no-playlist",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--max-downloads",
        str(max_downloads),
        "--match-filter",
        f"duration >= {max(1, min_duration_s)}",
        "-o",
        out_tpl,
        f"ytsearch{max_downloads}:{query}",
    ]
    res = _run(cmd, check=False)
    if res.returncode != 0 and not any(raw_dir.glob(f"{label}_yt_{source_id}_*.wav")):
        rows.append(
            {
                "label": label,
                "source_type": "youtube_query",
                "source_ref": query,
                "url": f"ytsearch:{query}",
                "raw_path": "",
                "processed_path": "",
                "status": "error",
                "error": (res.stderr or res.stdout or "yt-dlp failed")[:500],
            }
        )
        return rows

    for wav in raw_dir.glob(f"{label}_yt_{source_id}_*.wav"):
        rows.append(
            {
                "label": label,
                "source_type": "youtube_query",
                "source_ref": query,
                "url": "",
                "raw_path": str(wav),
                "processed_path": "",
                "status": "downloaded",
                "error": "",
            }
        )
    return rows


def _collect_youtube_url(
    url: str,
    label: str,
    raw_dir: Path,
    dry_run: bool,
) -> list[dict[str, str]]:
    source_id = _sha1(f"yt-url:{label}:{url}")
    if dry_run:
        return [
            {
                "label": label,
                "source_type": "youtube_url",
                "source_ref": "direct_url",
                "url": url,
                "raw_path": "",
                "processed_path": "",
                "status": "dry_run",
                "error": "",
            }
        ]

    out_tpl = str(raw_dir / f"{label}_yturl_{source_id}_%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--ignore-errors",
        "--no-playlist",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "-o",
        out_tpl,
        url,
    ]
    res = _run(cmd, check=False)
    rows = []
    files = list(raw_dir.glob(f"{label}_yturl_{source_id}_*.wav"))
    if res.returncode != 0 and not files:
        rows.append(
            {
                "label": label,
                "source_type": "youtube_url",
                "source_ref": "direct_url",
                "url": url,
                "raw_path": "",
                "processed_path": "",
                "status": "error",
                "error": (res.stderr or res.stdout or "yt-dlp failed")[:500],
            }
        )
        return rows
    for wav in files:
        rows.append(
            {
                "label": label,
                "source_type": "youtube_url",
                "source_ref": "direct_url",
                "url": url,
                "raw_path": str(wav),
                "processed_path": "",
                "status": "downloaded",
                "error": "",
            }
        )
    return rows


def _collect_podcast_feed(
    feed_url: str,
    label: str,
    raw_dir: Path,
    max_episodes: int,
    dry_run: bool,
) -> list[dict[str, str]]:
    rows = []
    source_id = _sha1(f"feed:{label}:{feed_url}")
    if dry_run:
        return [
            {
                "label": label,
                "source_type": "podcast_feed",
                "source_ref": feed_url,
                "url": "",
                "raw_path": "",
                "processed_path": "",
                "status": "dry_run",
                "error": "",
            }
        ]

    try:
        episodes = _parse_podcast_feed(feed_url, limit=max_episodes)
    except Exception as e:
        return [
            {
                "label": label,
                "source_type": "podcast_feed",
                "source_ref": feed_url,
                "url": "",
                "raw_path": "",
                "processed_path": "",
                "status": "error",
                "error": f"RSS parse failed: {str(e)[:400]}",
            }
        ]

    for idx, ep in enumerate(episodes, start=1):
        audio_url = ep["url"]
        title = ep["title"]
        stem = f"{label}_pod_{source_id}_{idx:03d}_{_slug(title, 24)}"
        raw_path = raw_dir / f"{stem}{_safe_ext_from_url(audio_url)}"
        ok, err = _download_url(audio_url, raw_path)
        rows.append(
            {
                "label": label,
                "source_type": "podcast_feed",
                "source_ref": feed_url,
                "url": audio_url,
                "raw_path": str(raw_path) if ok else "",
                "processed_path": "",
                "status": "downloaded" if ok else "error",
                "error": err,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "source_type",
        "source_ref",
        "url",
        "raw_path",
        "processed_path",
        "status",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect web audio for deepfake detector training")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/web_sources.json"),
        help="JSON config with real/fake source definitions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training/web_data"),
        help="Output root for raw/processed data and metadata",
    )
    parser.add_argument(
        "--max-youtube-per-query",
        type=int,
        default=5,
        help="Max downloads per YouTube query",
    )
    parser.add_argument(
        "--max-podcast-per-feed",
        type=int,
        default=4,
        help="Max podcast episodes per feed",
    )
    parser.add_argument(
        "--max-duration-s",
        type=int,
        default=45,
        help="Trim processed clips to this duration",
    )
    parser.add_argument(
        "--min-duration-s",
        type=int,
        default=20,
        help="Minimum YouTube duration filter for downloads",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output directory before collection",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not download; emit planned metadata only",
    )
    args = parser.parse_args()

    if not args.dry_run and not _ensure_tool("ffmpeg"):
        print("ERROR: ffmpeg is required.", file=sys.stderr)
        return 1

    if not args.dry_run and not _ensure_tool("yt-dlp"):
        print(
            "WARNING: yt-dlp not found. YouTube collection will be skipped. Install with: pip install yt-dlp",
            file=sys.stderr,
        )

    cfg = _load_config(args.config)
    out_root = args.output.resolve()
    raw_root = out_root / "raw"
    proc_root = out_root / "processed"
    metadata_path = out_root / "metadata.csv"

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    for label in ("real", "fake"):
        (raw_root / label).mkdir(parents=True, exist_ok=True)
        (proc_root / label).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for label in ("real", "fake"):
        label_cfg = cfg.get(label, {})
        raw_dir = raw_root / label

        if _ensure_tool("yt-dlp"):
            for q in label_cfg.get("youtube_queries", []) or []:
                rows.extend(
                    _collect_youtube_query(
                        query=q,
                        label=label,
                        raw_dir=raw_dir,
                        max_downloads=max(1, args.max_youtube_per_query),
                        min_duration_s=max(1, args.min_duration_s),
                        dry_run=args.dry_run,
                    )
                )
            for url in label_cfg.get("youtube_urls", []) or []:
                rows.extend(
                    _collect_youtube_url(
                        url=url,
                        label=label,
                        raw_dir=raw_dir,
                        dry_run=args.dry_run,
                    )
                )

        for feed in label_cfg.get("podcast_feeds", []) or []:
            rows.extend(
                _collect_podcast_feed(
                    feed_url=feed,
                    label=label,
                    raw_dir=raw_dir,
                    max_episodes=max(1, args.max_podcast_per_feed),
                    dry_run=args.dry_run,
                )
            )

    if not args.dry_run:
        # Normalize all successfully downloaded raw files.
        for row in rows:
            if row["status"] != "downloaded" or not row["raw_path"]:
                continue
            src = Path(row["raw_path"])
            if not src.exists():
                row["status"] = "error"
                row["error"] = "raw file missing"
                continue

            label = row["label"]
            digest = _sha1(str(src))
            out_name = f"{label}_{row['source_type']}_{digest}.wav"
            dst = proc_root / label / out_name
            ok, err = _normalize_audio(src, dst, max_duration_s=max(1, args.max_duration_s))
            if ok:
                row["processed_path"] = str(dst)
                row["status"] = "processed"
            else:
                row["status"] = "error"
                row["error"] = err

    _write_csv(metadata_path, rows)

    total = len(rows)
    processed = sum(1 for r in rows if r["status"] == "processed")
    downloaded = sum(1 for r in rows if r["status"] == "downloaded")
    errors = sum(1 for r in rows if r["status"] == "error")
    dry = sum(1 for r in rows if r["status"] == "dry_run")
    print(f"metadata: {metadata_path}")
    print(f"rows={total} processed={processed} downloaded={downloaded} dry_run={dry} errors={errors}")
    if errors > 0:
        print("Some items failed. Check metadata.csv error column.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

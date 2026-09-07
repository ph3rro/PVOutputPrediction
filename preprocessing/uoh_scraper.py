#!/usr/bin/env python3
"""
Scraper for the University of Hertfordshire AllSky camera video archive.

    https://observatory.herts.ac.uk/allsky/videoarchive.php

Discovers video links at runtime rather than assuming a fixed page layout, so it
should survive markup changes. Run with --dry-run first to see what it finds.

Usage:
    python allsky_scraper.py --dry-run
    python allsky_scraper.py --camera 1 --out ./allsky --delay 3
    python allsky_scraper.py --since 2026-01-01 --until 2026-03-31 --limit 50

Dependencies:
    pip install requests
    pip install beautifulsoup4   # optional; falls back to regex parsing
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import time
import urllib.parse
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path

import requests

try:
    from bs4 import BeautifulSoup
    HAVE_BS4 = True
except ImportError:
    HAVE_BS4 = False

BASE = "https://observatory.herts.ac.uk/allsky/"
ARCHIVE = urllib.parse.urljoin(BASE, "videoarchive.php")

# Set this to something that identifies you. Site operators are far more
# forgiving of a crawler they can email than an anonymous one.
USER_AGENT = "allsky-archive-fetcher/1.0 (research use; contact: you@example.edu)"

VIDEO_EXTS = (".mp4", ".webm", ".avi", ".mov", ".mpg", ".mpeg", ".m4v", ".mkv")

# Camera IDs used by the site's ?c= parameter (from the "Change camera" menu).
CAMERAS = {
    1: "bayfordbury",
    2: "hemel",
    3: "niton",
    4: "cromer",
    5: "guernsey",
    6: "exmoor",
    7: "bayfordbury-day",
}


@dataclass
class Video:
    url: str
    label: str
    date: dt.date | None

    @property
    def filename(self) -> str:
        path = Path(urllib.parse.urlparse(self.url).path)
        stem = path.name or re.sub(r"[^\w.-]+", "_", self.label) + ".mp4"
        # Video files are named only by date (2020-11-15.mp4), so prefix the
        # camera directory to keep different cameras from colliding.
        cam = next((p for p in path.parts if p.startswith("camera")), None)
        return f"{cam}_{stem}" if cam else stem


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

DATE_PATTERNS = [
    (re.compile(r"(20\d{2})[-_/](\d{2})[-_/](\d{2})"), lambda m: (m[1], m[2], m[3])),
    (re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"), lambda m: (m[1], m[2], m[3])),
]


def parse_date(text: str) -> dt.date | None:
    """Pull a YYYY-MM-DD or YYYYMMDD date out of a filename or link label."""
    for pattern, extract in DATE_PATTERNS:
        m = pattern.search(text)
        if m:
            try:
                y, mo, d = extract(m)
                return dt.date(int(y), int(mo), int(d))
            except ValueError:
                continue
    return None


def extract_links(html: str, page_url: str) -> list[tuple[str, str]]:
    """Return (absolute_url, link_text) pairs from a page."""
    pairs: list[tuple[str, str]] = []

    if HAVE_BS4:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(["a", "source", "video"]):
            href = tag.get("href") or tag.get("src")
            if href:
                pairs.append((urllib.parse.urljoin(page_url, href),
                              tag.get_text(strip=True) or href))
    else:
        for m in re.finditer(r'(?:href|src)\s*=\s*["\']([^"\']+)["\']', html, re.I):
            pairs.append((urllib.parse.urljoin(page_url, m.group(1)), m.group(1)))

    return pairs


def is_video(url: str) -> bool:
    path = urllib.parse.urlparse(url).path.lower()
    return path.endswith(VIDEO_EXTS)


def video_from_day_link(link: str) -> Video | None:
    """
    Monthly calendar pages link each day to video-new.php?d=DD&m=MM&y=YYYY&c=N.
    The day page just wraps a <video> whose source is the predictable URL
    camera{N}/videos/YYYY-MM-DD.mp4, so build that directly instead of
    fetching thousands of day pages.
    """
    p = urllib.parse.urlparse(link)
    if not p.path.endswith("video-new.php"):
        return None
    q = urllib.parse.parse_qs(p.query)
    try:
        d = int(q["d"][0])
        m = int(q["m"][0])
        y = int(q["y"][0])
        c = int(q.get("c", ["1"])[0])
    except (KeyError, ValueError, IndexError):
        return None
    if y < 100:
        y += 2000
    try:
        date = dt.date(y, m, d)
    except ValueError:
        return None
    url = urllib.parse.urljoin(BASE, f"camera{c}/videos/{date:%Y-%m-%d}.mp4")
    return Video(url, f"camera {c} {date:%Y-%m-%d}", date)


def _month_in_range(link: str, since: dt.date | None, until: dt.date | None) -> bool:
    """Skip monthly calendar pages (videos.php?m=&y=) outside the date filter."""
    p = urllib.parse.urlparse(link)
    if not p.path.endswith("videos.php"):
        return True
    q = urllib.parse.parse_qs(p.query)
    try:
        m = int(q["m"][0])
        y = int(q["y"][0])
    except (KeyError, ValueError, IndexError):
        return True
    if y < 100:
        y += 2000
    if since and (y, m) < (since.year, since.month):
        return False
    if until and (y, m) > (until.year, until.month):
        return False
    return True


def discover(session: requests.Session, camera: int | None,
             follow_depth: int = 1, delay: float = 2.0,
             since: dt.date | None = None,
             until: dt.date | None = None) -> list[Video]:
    """
    Fetch the archive index, follow the monthly calendar pages it links to,
    and build video URLs from each day's video-new.php link.
    """
    start = ARCHIVE if camera is None else f"{ARCHIVE}?c={camera}"
    seen_pages: set[str] = set()
    videos: dict[str, Video] = {}
    queue: list[tuple[str, int]] = [(start, 0)]

    while queue:
        url, depth = queue.pop(0)
        if url in seen_pages:
            continue
        seen_pages.add(url)

        print(f"  scanning {url}", file=sys.stderr)
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  ! failed: {exc}", file=sys.stderr)
            continue

        for link, label in extract_links(resp.text, url):
            if (day_video := video_from_day_link(link)) is not None:
                videos.setdefault(day_video.url, day_video)
            elif is_video(link):
                if link not in videos:
                    videos[link] = Video(link, label, parse_date(link) or parse_date(label))
            elif (depth < follow_depth and _is_same_archive(link, url)
                  and _month_in_range(link, since, until)):
                queue.append((link, depth + 1))

        time.sleep(delay)

    return sorted(videos.values(), key=lambda v: (v.date or dt.date.min, v.url))


def _is_same_archive(link: str, current: str) -> bool:
    """Only follow links that stay inside the /allsky/ area and look like pages."""
    p = urllib.parse.urlparse(link)
    if p.netloc and p.netloc != urllib.parse.urlparse(current).netloc:
        return False
    if "/allsky/" not in p.path:
        return False
    return p.path.endswith((".php", ".html", "/")) or "video" in p.query.lower()


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #

def download(session: requests.Session, video: Video, outdir: Path,
             delay: float, retries: int = 3) -> str:
    dest = outdir / video.filename
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists():
        try:
            head = session.head(video.url, timeout=20, allow_redirects=True)
            remote = int(head.headers.get("Content-Length", 0))
            if remote and remote == dest.stat().st_size:
                return "skipped (already complete)"
        except requests.RequestException:
            return "skipped (exists)"

    for attempt in range(1, retries + 1):
        try:
            with session.get(video.url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                written = 0
                with open(part, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)
                        written += len(chunk)
                        if total:
                            pct = 100 * written / total
                            print(f"\r    {dest.name}  {pct:5.1f}%",
                                  end="", file=sys.stderr)
                print("", file=sys.stderr)
            part.rename(dest)
            time.sleep(delay)
            return f"downloaded ({written / 1e6:.1f} MB)"
        except requests.HTTPError as exc:
            # The calendar links every day of the month, but not every night
            # has a video; a 4xx will not fix itself, so don't retry.
            status = exc.response.status_code if exc.response is not None else 0
            if 400 <= status < 500:
                part.unlink(missing_ok=True)
                time.sleep(delay)
                return f"not available (HTTP {status})"
            wait = 2 ** attempt
            print(f"    attempt {attempt} failed ({exc}); retrying in {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"    attempt {attempt} failed ({exc}); retrying in {wait}s",
                  file=sys.stderr)
            time.sleep(wait)

    part.unlink(missing_ok=True)
    return "FAILED"


# --------------------------------------------------------------------------- #
# Politeness
# --------------------------------------------------------------------------- #

def check_robots(session: requests.Session) -> bool:
    """Returns True if robots.txt permits fetching the archive."""
    rp = urllib.robotparser.RobotFileParser()
    robots_url = urllib.parse.urljoin(BASE, "/robots.txt")
    try:
        resp = session.get(robots_url, timeout=15)
        if resp.status_code >= 400:
            return True  # no robots.txt served
        rp.parse(resp.text.splitlines())
    except requests.RequestException:
        return True
    return rp.can_fetch(USER_AGENT, ARCHIVE)


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, choices=sorted(CAMERAS),
                    help="camera id (?c=); omit for the default archive view")
    ap.add_argument("--out", type=Path, default=Path("allsky_videos"))
    ap.add_argument("--delay", type=float, default=2.0,
                    help="seconds between requests (default 2)")
    ap.add_argument("--depth", type=int, default=1,
                    help="how many link levels to follow from the index")
    ap.add_argument("--since", type=dt.date.fromisoformat)
    ap.add_argument("--until", type=dt.date.fromisoformat)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be downloaded, then exit")
    ap.add_argument("--ignore-robots", action="store_true")
    args = ap.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    if not args.ignore_robots and not check_robots(session):
        print("robots.txt disallows automated access to this path.\n"
              "Contact the observatory for archive access, or pass "
              "--ignore-robots if you already have permission.", file=sys.stderr)
        return 2

    print("Discovering videos...", file=sys.stderr)
    videos = discover(session, args.camera, args.depth, args.delay,
                      args.since, args.until)

    if args.since:
        videos = [v for v in videos if v.date and v.date >= args.since]
    if args.until:
        videos = [v for v in videos if v.date and v.date <= args.until]
    if args.limit:
        videos = videos[:args.limit]

    if not videos:
        print("No video links found. The archive may load its listing via "
              "JavaScript or a POST form — inspect the page in a browser's "
              "network tab and adjust discover() accordingly.", file=sys.stderr)
        return 1

    print(f"\nFound {len(videos)} video(s):", file=sys.stderr)
    for v in videos:
        print(f"  {v.date or '????-??-??'}  {v.filename}")

    if args.dry_run:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading to {args.out.resolve()}\n", file=sys.stderr)
    for i, v in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {v.filename}", file=sys.stderr)
        print(f"    {download(session, v, args.out, args.delay)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

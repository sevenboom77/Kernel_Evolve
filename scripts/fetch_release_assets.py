"""Fetch optional, SHA256-bound datasets and historical AGH source archives."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fetch(record: dict, manifest: dict) -> None:
    target = (ROOT / record["target"]).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError("Asset destination escapes repository")
    if target.exists():
        if digest(target) != record["sha256"]:
            raise ValueError(f"Existing file differs from manifest; move it aside: {target}")
        print(f"Verified existing: {record['target']}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://github.com/{manifest['repository']}/releases/download/"
        f"{manifest['release']}/{record['asset']}"
    )
    print(f"Downloading {record['asset']} ({record['download_bytes']:,} bytes)")
    with tempfile.TemporaryDirectory(prefix=".kernel-evolve-download-", dir=target.parent) as folder:
        download = Path(folder) / "download"
        request = Request(url, headers={"User-Agent": "KernelEvolve-asset-fetcher"})
        with urlopen(request, timeout=120) as response, download.open("wb") as output:
            shutil.copyfileobj(response, output)
        if download.stat().st_size != record["download_bytes"]:
            raise ValueError(f"Download size mismatch: {record['asset']}")
        if digest(download) != record["download_sha256"]:
            raise ValueError(f"Download SHA256 mismatch: {record['asset']}")
        unpacked = Path(folder) / "verified"
        if record["compression"] == "gzip":
            with gzip.open(download, "rb") as source, unpacked.open("wb") as output:
                shutil.copyfileobj(source, output)
        elif record["compression"] == "none":
            download.rename(unpacked)
        else:
            raise ValueError("Unsupported asset compression")
        if unpacked.stat().st_size != record["size_bytes"] or digest(unpacked) != record["sha256"]:
            raise ValueError(f"Asset SHA256 mismatch: {record['target']}")
        if target.exists():
            raise FileExistsError(f"Refusing to replace a newly created file: {target}")
        unpacked.rename(target)
    print(f"Ready: {record['target']}")


def main() -> None:
    manifest = json.loads((ROOT / "release-assets.json").read_text(encoding="utf-8"))
    identifiers = [record["id"] for record in manifest["assets"]]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="store_true", help="Fetch the original ogbn-arxiv archive")
    parser.add_argument("--sources", action="store_true", help="Fetch all three historical source.tar files")
    parser.add_argument("--all", action="store_true", help="Fetch every asset")
    parser.add_argument("--asset", action="append", choices=identifiers, default=[])
    parser.add_argument("--list", action="store_true", help="List assets without downloading")
    args = parser.parse_args()
    if args.list:
        for record in manifest["assets"]:
            print(f"{record['id']:12} {record['target']:28} {record['download_bytes']:,} bytes")
        return
    chosen = set(args.asset)
    if args.dataset:
        chosen.add("arxiv")
    if args.sources:
        chosen.update(identifier for identifier in identifiers if identifier.startswith("agh-"))
    if args.all:
        chosen.update(identifiers)
    if not chosen:
        parser.error("Choose --dataset, --sources, --all, --asset or --list")
    for record in manifest["assets"]:
        if record["id"] in chosen:
            fetch(record, manifest)


if __name__ == "__main__":
    main()

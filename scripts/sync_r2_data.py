from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _build_client():
    endpoint = _env("R2_ENDPOINT")
    access_key = _env("R2_ACCESS_KEY_ID")
    secret_key = _env("R2_SECRET_ACCESS_KEY")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 10, "mode": "standard"},
        ),
    )


def _iter_objects(client, bucket: str, prefix: str):
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if key and not key.endswith("/"):
                yield key, int(obj.get("Size", 0))


def _needs_download(local_path: Path, remote_size: int) -> bool:
    try:
        return local_path.stat().st_size != remote_size
    except FileNotFoundError:
        return True


def _download_tree(client, bucket: str, prefix: str, dest_root: Path) -> tuple[int, int]:
    downloaded = 0
    skipped = 0
    for key, size in _iter_objects(client, bucket, prefix):
        local_path = dest_root / Path(key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not _needs_download(local_path, size):
            skipped += 1
            continue
        print(f"Downloading {key} -> {local_path}", flush=True)
        client.download_file(bucket, key, str(local_path))
        downloaded += 1
    return downloaded, skipped


def main() -> int:
    bucket = _env("R2_BUCKET")
    dest_root = Path(os.getenv("HYDROVISION_DATA_DIR", "/tmp/hydrovision-data")).expanduser()
    prefixes = [p.strip().strip("/") for p in os.getenv("R2_SYNC_PREFIXES", "Mekong,LamaH").split(",") if p.strip()]

    dest_root.mkdir(parents=True, exist_ok=True)
    client = _build_client()

    total_downloaded = 0
    total_skipped = 0
    for prefix in prefixes:
        # If prefix looks like a file (has an extension), download it directly.
        # Otherwise treat it as a directory prefix and recurse.
        if '.' in prefix.split('/')[-1]:
            local_path = dest_root / Path(prefix)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                head = client.head_object(Bucket=bucket, Key=prefix)
                remote_size = int(head.get('ContentLength', -1))
                if _needs_download(local_path, remote_size):
                    print(f"Downloading {prefix} -> {local_path}", flush=True)
                    client.download_file(bucket, prefix, str(local_path))
                    total_downloaded += 1
                else:
                    total_skipped += 1
            except Exception as e:
                print(f"Warning: could not download {prefix}: {e}", flush=True)
        else:
            downloaded, skipped = _download_tree(client, bucket, f"{prefix}/", dest_root)
            total_downloaded += downloaded
            total_skipped += skipped

    print(
        f"R2 sync complete. Downloaded {total_downloaded} files, skipped {total_skipped} unchanged files.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"R2 sync failed: {exc}", file=sys.stderr, flush=True)
        raise

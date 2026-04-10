"""Compression vs move: distinguish gzip/tar from mv."""

NAME = "Compression"
TAGS = ["compress", "gzip", "tar", "zip", "archive", "compression"]
TRIGGERS = [
    r"compress", r"\bgzip\b", r"\btar\b", r"\bzip\b", r"archive",
    r"\.gz\b", r"\.tar\b", r"\.zip\b", r"\.tgz\b",
]
ALWAYS = False

CONTENT = """\
Compression techniques:
- "Compress" means reducing file size, NOT moving. Common tools:
  - `gzip file.txt` → produces `file.txt.gz` (replaces original)
  - `tar -czf archive.tar.gz dir/` → tarball with gzip compression
  - `zip -r archive.zip dir/` → zip archive
- A "compressed" file should have a `.gz`, `.tar.gz`, `.zip`, etc. extension.
- If the task says "compress files older than 30 days and move them", you must actually run gzip/tar — `mv` alone is not compression.
- For batches: `find /path -mtime +30 -exec gzip {} \\;` compresses files older than 30 days in place.\
"""

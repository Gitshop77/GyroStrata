"""Check release-file SHA256s before installation; no dependencies or downloads."""

import hashlib, pathlib, re, sys


def verify_release(root=None):
    root = pathlib.Path(root or pathlib.Path(__file__).resolve().parent).resolve()
    seen = set()
    for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8-sig").splitlines():
        digest, relative = line.split("  ", 1)
        path = root / relative
        if (
            relative in seen
            or not re.fullmatch(r"[a-f0-9]{64}", digest)
            or not path.resolve().is_relative_to(root)
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise RuntimeError(
                "Release file differs from SHA256SUMS.txt: "
                + relative
                + ". Extract a fresh copy of this release."
            )
        seen.add(relative)
    if not seen:
        raise RuntimeError("Release checksum manifest is empty.")
    return len(seen)


if __name__ == "__main__":
    try:
        print("PASS:", verify_release(), "release files match SHA256SUMS.txt")
    except (OSError, ValueError, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)

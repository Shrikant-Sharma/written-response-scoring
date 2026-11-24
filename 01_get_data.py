"""Step 1. Get the one data file and check it against its fingerprint.

    python 01_get_data.py            download it if it is missing (16 MB), then check it
    python 01_get_data.py --check    only check the file already in data/raw/

The data is the training file of the Automated Student Assessment Prize (2012): 12,976 written responses on 8
prompts, each scored by two trained human raters. See data/manifest.json for the source and the license note.
"""
import sys
import urllib.request

from common import DATA_FILE, RAW, ROOT, read_json, sha256_of


def main():
    entry = read_json(ROOT / "data" / "manifest.json")["file"]
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / DATA_FILE
    if not path.exists():
        if "--check" in sys.argv:
            sys.exit(f"{path} is missing. Run without --check to download it.")
        print(f"downloading {DATA_FILE} ...", flush=True)
        request = urllib.request.Request(entry["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response, open(path, "wb") as out:
            out.write(response.read())
    found = sha256_of(path)
    if found != entry["sha256"]:
        sys.exit(f"{DATA_FILE} does not match the manifest.\n  expected {entry['sha256']}\n  found    {found}")
    print(f"ok  {DATA_FILE}  ({path.stat().st_size:,} bytes). The file matches the manifest.")


if __name__ == "__main__":
    main()

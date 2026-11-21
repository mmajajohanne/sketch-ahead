"""Download Quick, Draw! simplified ndjson files for the configured classes.

Usage (from train/):
    python -m data.download            # all classes in config.CLASSES
    python -m data.download cat dog    # only some classes

Files land in train/raw/<class>.ndjson (gitignored). Each file is roughly
50–200 MB. Data: Google, CC BY 4.0.
"""

import sys
import urllib.parse
import urllib.request
from pathlib import Path

from config import CLASSES

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
RAW_DIR = Path(__file__).resolve().parent.parent / "raw"


def download(name: str, out_dir: Path = RAW_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.ndjson"
    if out.exists():
        print(f"[skip] {out.name} already exists")
        return out
    url = BASE_URL + urllib.parse.quote(f"{name}.ndjson")
    tmp = out.with_suffix(".part")
    print(f"[get]  {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(out)
    return out


if __name__ == "__main__":
    for name in sys.argv[1:] or CLASSES:
        download(name)

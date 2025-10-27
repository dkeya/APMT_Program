import sys
from pathlib import Path

# Uses geoBoundaries open data (ADM1 = counties, ADM2 = sub-counties)
URLS = {
    "counties": "https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/KEN/ADM1/geoBoundaries-KEN-ADM1.geojson",
    "subcounties": "https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/KEN/ADM2/geoBoundaries-KEN-ADM2.geojson",
}

def fetch(url: str, out_path: Path):
    try:
        import requests  # lazy import so we can show a clearer error if missing
    except ImportError:
        print("The 'requests' package is required. Install with: py -3 -m pip install requests")
        sys.exit(1)

    print(f"Downloading {url} ...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    print(f"Saved -> {out_path} ({len(r.content):,} bytes)")

def main():
    out_dir = Path("geo")
    out_dir.mkdir(parents=True, exist_ok=True)

    fetch(URLS["counties"], out_dir / "kenya_counties.geojson")
    fetch(URLS["subcounties"], out_dir / "kenya_subcounties.geojson")
    print("All done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Failed:", e)
        sys.exit(1)

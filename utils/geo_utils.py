# apmt_dashboard/utils/geo_utils.py
import os
import requests
from pathlib import Path

def ensure_geo_assets():
    """Ensure county/sub-county GeoJSON files exist (auto-download if missing)."""
    import os, requests
    os.makedirs("geo", exist_ok=True)
    assets = {
        "geo/kenya_counties.geojson":    "https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/KEN/ADM1/geoBoundaries-KEN-ADM1.geojson",
        "geo/kenya_subcounties.geojson": "https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/gbOpen/KEN/ADM2/geoBoundaries-KEN-ADM2.geojson",
    }

    missing = [p for p in assets if not (os.path.exists(p) and os.path.getsize(p) > 0)]
    if not missing:
        return True  # already present

    for path in missing:
        url = assets[path]
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            print(f"Could not fetch {os.path.basename(path)}: {e}")
            return False
    return True
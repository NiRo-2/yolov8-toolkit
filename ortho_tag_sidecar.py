"""
Helpers so detection JSON sidecars work with Ortho-Tag / B3DM (`detect_to_3d`).

Ortho-Tag reads `metadata` keys in exiftool -G1 style (e.g. `GPS:GPSLatitude`,
`XMP-drone-dji:FlightYawDegree`). Pillow's flat `getexif()` dict uses different
names; this module can merge GPS (and basic Exif IFD focal) into those keys when
exiftool was not used.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Optional: Pillow only used for merge path
try:
    from PIL import Image, ExifTags
    from PIL.ExifTags import IFD
except ImportError:
    Image = None  # type: ignore
    ExifTags = None  # type: ignore
    IFD = None  # type: ignore


def _rational_to_float(x: Any) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if hasattr(x, "numerator") and hasattr(x, "denominator"):
        d = float(x.denominator)
        return float(x.numerator) / d if d else 0.0
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return float(x[0]) / float(x[1]) if float(x[1]) != 0 else 0.0
    return float(x)


def _gps_coord_degrees(values: Any, ref: Optional[str]) -> Optional[float]:
    if not values or not isinstance(values, (tuple, list)) or len(values) < 3:
        return None
    d = _rational_to_float(values[0])
    m = _rational_to_float(values[1])
    s = _rational_to_float(values[2])
    dec = d + m / 60.0 + s / 3600.0
    ref_u = (ref or "").strip().upper()[:1]
    if ref_u in ("S", "W"):
        dec = -dec
    return dec


def _has_exiftool_style_pose_keys(metadata: Dict[str, Any]) -> bool:
    """True if metadata already looks like exiftool -G1 output for pose/GPS."""
    for k in metadata:
        if not isinstance(k, str):
            continue
        if k.startswith("GPS:") or k.startswith("XMP-drone-dji:"):
            return True
    return False


def merge_pillow_gps_exif_into_metadata(
    metadata: Dict[str, Any],
    img_path: Path,
) -> Dict[str, Any]:
    """
    Copy metadata and add GPS:* / ExifIFD:FocalLength-style keys from Pillow IFDs
    when exiftool-style keys are absent.
    """
    out = dict(metadata)
    if Image is None or IFD is None:
        return out
    if _has_exiftool_style_pose_keys(out):
        return out
    try:
        with Image.open(img_path) as im:
            exif = im.getexif()
            if not exif:
                return out
            gps = exif.get_ifd(IFD.GPSInfo)
            if isinstance(gps, dict) and gps:
                lat = _gps_coord_degrees(
                    gps.get(2),
                    gps.get(1) if isinstance(gps.get(1), str) else None,
                )
                lon = _gps_coord_degrees(
                    gps.get(4),
                    gps.get(3) if isinstance(gps.get(3), str) else None,
                )
                if lat is not None:
                    out["GPS:GPSLatitude"] = lat
                if lon is not None:
                    out["GPS:GPSLongitude"] = lon
                alt = gps.get(6)
                if alt is not None:
                    out["GPS:GPSAltitude"] = _rational_to_float(alt)
                img_dir = gps.get(17)
                if img_dir is not None:
                    out["GPS:GPSImgDirection"] = _rational_to_float(img_dir)
            exif_ifd = exif.get_ifd(IFD.Exif)
            if isinstance(exif_ifd, dict) and exif_ifd:
                # 0x920A FocalLength, 0xA405 FocalLengthIn35mmFilm
                fl = exif_ifd.get(0x920A)
                if fl is not None:
                    out["ExifIFD:FocalLength"] = _rational_to_float(fl)
                f35 = exif_ifd.get(0xA405)
                if f35 is not None:
                    out["ExifIFD:FocalLengthIn35mmFormat"] = _rational_to_float(f35)
                iw = exif_ifd.get(0xA002)
                ih = exif_ifd.get(0xA003)
                if iw is not None:
                    out["File:ImageWidth"] = int(iw)
                if ih is not None:
                    out["File:ImageHeight"] = int(ih)
    except Exception:
        return out
    return out


# Ortho-Tag `extract_camera_from_json_payload` requires lat/lon; other fields improve pose/FOV.
B3DM_CRITICAL_CHECKS: List[Tuple[str, Callable[[Dict[str, Any]], Any]]] = [
    (
        "latitude (GPS:GPSLatitude or XMP-drone-dji:GPSLatitude)",
        lambda m: m.get("GPS:GPSLatitude") or m.get("XMP-drone-dji:GPSLatitude"),
    ),
    (
        "longitude (GPS:GPSLongitude or XMP-drone-dji:GPSLongitude)",
        lambda m: m.get("GPS:GPSLongitude") or m.get("XMP-drone-dji:GPSLongitude"),
    ),
]

B3DM_RECOMMENDED_CHECKS: List[Tuple[str, Callable[[Dict[str, Any]], Any]]] = [
    (
        "altitude (RelativeAltitude / AbsoluteAltitude / GPSAltitude)",
        lambda m: m.get("XMP-drone-dji:RelativeAltitude")
        or m.get("XMP-drone-dji:AbsoluteAltitude")
        or m.get("GPS:GPSAltitude"),
    ),
    (
        "heading (GPSImgDirection or FlightYawDegree)",
        lambda m: m.get("GPS:GPSImgDirection") or m.get("XMP-drone-dji:FlightYawDegree"),
    ),
    (
        "gimbal pitch (GimbalPitchDegree)",
        lambda m: m.get("XMP-drone-dji:GimbalPitchDegree"),
    ),
    (
        "focal length (ExifIFD:FocalLength)",
        lambda m: m.get("ExifIFD:FocalLength"),
    ),
]


def verify_b3dm_sidecar_metadata(metadata: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Return (critical_ok, missing_critical, missing_recommended) for Ortho-Tag sidecars.
    """
    miss_c: List[str] = []
    for label, getter in B3DM_CRITICAL_CHECKS:
        try:
            v = getter(metadata)
        except Exception:
            v = None
        if v is None or (isinstance(v, str) and not str(v).strip()):
            miss_c.append(label)
    miss_r: List[str] = []
    for label, getter in B3DM_RECOMMENDED_CHECKS:
        try:
            v = getter(metadata)
        except Exception:
            v = None
        if v is None or (isinstance(v, str) and not str(v).strip()):
            miss_r.append(label)
    return (len(miss_c) == 0, miss_c, miss_r)


def verify_sidecar_json_file(path: Path) -> Tuple[bool, List[str], List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("metadata") if isinstance(data, dict) else None
    if not isinstance(meta, dict):
        return False, ["metadata object missing or not a dict"], []
    ok_c, miss_c, miss_r = verify_b3dm_sidecar_metadata(meta)
    return ok_c, miss_c, miss_r


def main_verify(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("Usage: python ortho_tag_sidecar.py <sidecar.json>", file=sys.stderr)
        return 2
    p = Path(argv[0])
    if not p.is_file():
        print(f"Not a file: {p}", file=sys.stderr)
        return 2
    ok_c, miss_c, miss_r = verify_sidecar_json_file(p)
    if miss_r is None:
        miss_r = []
    if ok_c:
        print(f"OK (georeference): {p} has latitude and longitude in exiftool-style keys.")
    else:
        print(f"FAIL (georeference): {p}")
        for m in miss_c:
            print(f"  - {m}")
    if miss_r:
        print("Recommended for FOV / pose (often needs exiftool for DJI XMP):")
        for m in miss_r:
            print(f"  - {m}")
    if not ok_c:
        print(
            "\nTip: run detect_images.py with exiftool on PATH or --exiftool so "
            "metadata uses -G1 keys (XMP-drone-dji:*, GPS:*, ExifIFD:*)."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main_verify())

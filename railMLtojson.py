# preprocess_railml3.py  — run locally, commit the output infrastructure.json
"""
Preprocessor for Czech national railML 3.2 infrastructure files.

Usage:
    python preprocess_railml3.py input.xml [output.json]

Output:
    A compact JSON file (~1–5 MB vs 98 MB source) containing one entry per
    linearPositioningSystem (= one named track segment), ready for the
    Streamlit app's track selector dropdown.
"""

import xml.etree.ElementTree as ET
import json
import sys
import os
from collections import defaultdict


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def strip_ns(root):
    """Remove XML namespaces from all tags so we can query by bare tag name."""
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
        # Also strip namespace prefixes from attribute keys
        stripped = {}
        for k, v in elem.attrib.items():
            if '}' in k:
                k = k.split('}', 1)[1]
            stripped[k] = v
        elem.attrib = stripped
    return root


def m_to_km(val):
    return float(val) / 1000.0


# ─────────────────────────────────────────────
#  Step 1 — Build LPS (track) index
# ─────────────────────────────────────────────

def extract_lps_index(root):
    """
    Returns dict: lps_id → {"label": "...", "start_m": float, "end_m": float}

    linearPositioningSystem lives under:
      railML / common / positioning / linearPositioningSystems / linearPositioningSystem
    """
    lps = {}
    for el in root.iter('linearPositioningSystem'):
        lps_id = el.get('id', '')
        start  = el.get('startMeasure')
        end    = el.get('endMeasure')
        # Name is in a <name> child element
        name_el = el.find('name')
        label   = name_el.get('name', lps_id) if name_el is not None else lps_id

        if lps_id:
            lps[lps_id] = {
                "label":   label,
                "start_m": float(start) if start else 0.0,
                "end_m":   float(end)   if end   else 0.0,
            }

    print(f"  Found {len(lps)} linearPositioningSystems (track segments)")
    return lps


# ─────────────────────────────────────────────
#  Step 2 — Build a position lookup for every element
#  that carries a linearCoordinate
# ─────────────────────────────────────────────

def build_position_index(root):
    """
    Walk the entire tree and collect every element that has at least one
    <linearCoordinate positioningSystemRef="..." measure="..."/>.

    Returns:
        pos_index: dict
            lps_id → list of {"elem": Element, "measure_m": float}

    We index by the *parent* of the linearCoordinate so we can read
    sibling attributes (speed value, gradient value, station name, etc.)
    """
    pos_index = defaultdict(list)

    for lc in root.iter('linearCoordinate'):
        lps_ref  = lc.get('positioningSystemRef', '')
        measure  = lc.get('measure')
        if not (lps_ref and measure):
            continue

        # Walk up to find the meaningful ancestor carrying domain data.
        # We store the intrinsicCoordinate's parent chain up to 4 levels.
        # The actual data (speed, gradient, name) usually sits 2–4 levels up.
        parent = lc
        for _ in range(6):
            parent = _parent_map.get(id(parent))
            if parent is None:
                break
            tag = parent.tag
            # Stop at elements that are likely domain containers
            if tag in ('operationalPoint', 'speedSection', 'gradient',
                       'trackCondition', 'netElement', 'line', 'border',
                       'levelCrossing', 'bridge', 'tunnel', 'signal',
                       'trainDetectionElement', 'derailer', 'switch',
                       'crossing', 'bufferstop', 'openEnd', 'connection'):
                pos_index[lps_ref].append({
                    "elem":      parent,
                    "measure_m": float(measure)
                })
                break

    return pos_index


# ─────────────────────────────────────────────
#  Step 3 — Extract operational points (stations)
# ─────────────────────────────────────────────

def extract_stations(root, lps_ids: set):
    """
    operationalPoint elements carry the station name and a position.
    In railML 3.2 they live under:
      railML / infrastructure / functionalInfrastructure / operationalPoints / operationalPoint

    Each has one or more <spotElementProjection> children that contain
    <linearCoordinate> references.

    Returns: list of {"lps_id": ..., "measure_m": ..., "name": ..., "stop_type": "X"/"R"}
    """
    stations = []
    for op in root.iter('operationalPoint'):
        op_id   = op.get('id', '')
        name_el = op.find('name')
        name    = name_el.get('name', op_id) if name_el is not None else op_id

        # Determine whether it is a mandatory stop or request stop.
        # railML 3.2 uses <isStation>, <isHalt>, <isJunction>, etc.
        stop_type = 'R'  # default: request / optional
        if op.find('isStation') is not None:
            stop_type = 'X'
        elif op.find('isHalt') is not None:
            stop_type = 'X'  # halts are still passenger stops

        # Collect positions on each LPS this OP touches
        for lc in op.iter('linearCoordinate'):
            lps_ref = lc.get('positioningSystemRef', '')
            measure = lc.get('measure')
            if lps_ref in lps_ids and measure:
                stations.append({
                    "lps_id":    lps_ref,
                    "measure_m": float(measure),
                    "name":      name,
                    "stop_type": stop_type
                })

    print(f"  Found {len(stations)} station position references")
    return stations


# ─────────────────────────────────────────────
#  Step 4 — Extract speed sections
# ─────────────────────────────────────────────

def extract_speeds(root, lps_ids: set):
    """
    speedSection elements live under:
      railML / infrastructure / functionalInfrastructure / speeds / speedSection

    Each has a <speedProfileRef> and start/end positions via linearCoordinates.
    The speed value is in attribute `vMax` (km/h) on the speedSection itself,
    or inside a nested <speedProfileRef>.

    Returns: list of {"lps_id": ..., "measure_m": ..., "speed_kmh": float}
    """
    speeds = []
    for ss in root.iter('speedSection'):
        v_max = ss.get('vMax') or ss.get('v') or ss.get('maxSpeed')
        if v_max is None:
            # Try child elements
            spr = ss.find('speedProfileRef')
            if spr is not None:
                v_max = spr.get('vMax') or spr.get('v')
        if v_max is None:
            continue

        for lc in ss.iter('linearCoordinate'):
            lps_ref = lc.get('positioningSystemRef', '')
            measure = lc.get('measure')
            if lps_ref in lps_ids and measure:
                speeds.append({
                    "lps_id":    lps_ref,
                    "measure_m": float(measure),
                    "speed_kmh": float(v_max)
                })

    print(f"  Found {len(speeds)} speed position references")
    return speeds


# ─────────────────────────────────────────────
#  Step 5 — Extract gradients
# ─────────────────────────────────────────────

def extract_gradients(root, lps_ids: set):
    """
    Gradient (slope) information in railML 3.2 lives under:
      railML / infrastructure / functionalInfrastructure / trackConditions / trackCondition
      or under <gradient> elements directly.

    The slope value is in ‰ (per mille), stored in attribute `slope` or `grade`.

    Returns: list of {"lps_id": ..., "measure_m": ..., "slope_pm": float}
    """
    gradients = []

    # Try <gradient> direct children first (railML 3.x often nests these)
    for grad in root.iter('gradient'):
        slope = grad.get('slope') or grad.get('grade') or grad.get('value')
        if slope is None:
            continue
        for lc in grad.iter('linearCoordinate'):
            lps_ref = lc.get('positioningSystemRef', '')
            measure = lc.get('measure')
            if lps_ref in lps_ids and measure:
                gradients.append({
                    "lps_id":   lps_ref,
                    "measure_m": float(measure),
                    "slope_pm": float(slope)
                })

    # Also check <trackCondition> wrappers (some 3.2 profiles use this)
    for tc in root.iter('trackCondition'):
        slope = tc.get('slope') or tc.get('grade')
        if slope is None:
            continue
        for lc in tc.iter('linearCoordinate'):
            lps_ref = lc.get('positioningSystemRef', '')
            measure = lc.get('measure')
            if lps_ref in lps_ids and measure:
                gradients.append({
                    "lps_id":   lps_ref,
                    "measure_m": float(measure),
                    "slope_pm": float(slope)
                })

    print(f"  Found {len(gradients)} gradient position references")
    return gradients


# ─────────────────────────────────────────────
#  Step 6 — Assemble per-track event lists
# ─────────────────────────────────────────────

def assemble_tracks(lps_index, stations, speeds, gradients):
    """
    Merge all events per LPS into a sorted event list compatible with the
    simulator's standard DataFrame format.

    Returns: dict  lps_id → {"label": ..., "events": [...]}
    """
    tracks = {}

    for lps_id, meta in lps_index.items():
        events = []

        # Station events
        for s in stations:
            if s["lps_id"] == lps_id:
                events.append({
                    "pos":       m_to_km(s["measure_m"]),
                    "name":      s["name"],
                    "stop_type": s["stop_type"],
                    "speed":     None,
                    "gradient":  None
                })

        # Speed events
        for sp in speeds:
            if sp["lps_id"] == lps_id:
                events.append({
                    "pos":       m_to_km(sp["measure_m"]),
                    "name":      "",
                    "stop_type": "",
                    "speed":     sp["speed_kmh"],
                    "gradient":  None
                })

        # Gradient events
        for g in gradients:
            if g["lps_id"] == lps_id:
                events.append({
                    "pos":       m_to_km(g["measure_m"]),
                    "name":      "",
                    "stop_type": "",
                    "speed":     None,
                    "gradient":  g["slope_pm"]
                })

        if not events:
            continue  # Skip LPS segments with no usable data

        events.sort(key=lambda e: e["pos"])
        tracks[lps_id] = {
            "label":    meta["label"],
            "start_km": m_to_km(meta["start_m"]),
            "end_km":   m_to_km(meta["end_m"]),
            "events":   events
        }

    return tracks


# ─────────────────────────────────────────────
#  Diagnostics — print a sample to verify
# ─────────────────────────────────────────────

def print_sample(tracks, n=5):
    print(f"\n── Sample output ({min(n, len(tracks))} of {len(tracks)} tracks) ──")
    for tid, t in list(tracks.items())[:n]:
        stations_in = [e for e in t["events"] if e["stop_type"] in ("X", "R")]
        speeds_in   = [e for e in t["events"] if e["speed"] is not None]
        grads_in    = [e for e in t["events"] if e["gradient"] is not None]
        print(f"  [{tid}]  \"{t['label']}\"")
        print(f"    Range : {t['start_km']:.3f} – {t['end_km']:.3f} km")
        print(f"    Events: {len(stations_in)} stations | {len(speeds_in)} speed pts | {len(grads_in)} gradient pts")
        if stations_in:
            names = [s["name"] for s in stations_in[:4]]
            print(f"    Stops : {', '.join(names)}{'…' if len(stations_in) > 4 else ''}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

_parent_map = {}   # id(child) → parent Element, populated in main()

def main(src: str, dst: str):
    print(f"\nParsing {src}  ({os.path.getsize(src)/1e6:.1f} MB)…")
    tree = ET.parse(src)
    root = strip_ns(tree.getroot())

    # Build a global child→parent map so position_index can walk upward
    global _parent_map
    _parent_map = {id(child): parent for parent in root.iter() for child in parent}

    print("Extracting linearPositioningSystems…")
    lps_index = extract_lps_index(root)
    lps_ids   = set(lps_index.keys())

    print("Extracting operational points (stations)…")
    stations  = extract_stations(root, lps_ids)

    print("Extracting speed sections…")
    speeds    = extract_speeds(root, lps_ids)

    print("Extracting gradients…")
    gradients = extract_gradients(root, lps_ids)

    print("Assembling per-track event lists…")
    tracks    = assemble_tracks(lps_index, stations, speeds, gradients)

    print_sample(tracks)

    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(tracks, f, ensure_ascii=False, separators=(',', ':'))

    size_kb = os.path.getsize(dst) / 1024
    print(f"\n✅  {len(tracks)} tracks → {dst}  ({size_kb:.1f} KB)\n")


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else "railML_export_20251219_081227.xml"  # ← here
    dst = sys.argv[2] if len(sys.argv) > 2 else "infrastructure.json"
    main(src, dst)
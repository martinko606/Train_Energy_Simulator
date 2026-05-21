"""
DYPOD Track Profile Builder & Fleet Energy Simulator
=====================================================
Reads Czech railway infrastructure from a DYPOD railML 3.2 export
(Správa železnic / OLTIS Group), builds a direction-aware stitched track
profile between any two stations, then runs a physics-based energy
simulation with Monte Carlo stochastic analysis.

Data model follows DYPOD popis datových struktur v1.4:
  operationalPoint  → stations / stopping points (meso graph nodes)
  line              → segment between two OPs (meso graph edge)
                      beginsInOP / endsInOP / length / netElementRef
  gradientCurve     → average gradient [‰] per segment, normal / reverse direction
  speedSection      → max speed [km/h] per track sub-element, normal / reverse
  electrificationSection → electrification system per track sub-element
  netElement hierarchy   → segment ne_id contains track sub-element ids
                           via elementCollectionUnordered

Usage:
    streamlit run dypod_simulator.py
Then upload the railML zip or xml via the sidebar.
"""

from __future__ import annotations

import glob
import heapq
import itertools
import math
import os
import random
import re
import tempfile
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PASSENGER_STOP_TYPES = {"station", "stoppingPoint"}
TOPOLOGY_TYPES       = {"junction", "block", "siding", "borderPoint",
                         "crossover", "depot"}

LINE_CATEGORY_LABELS = {
    "other:KoridorovéCelostát":   "Corridor main",
    "other:HlavTahyCelostátní":   "Main line",
    "other:OstatníCelostátní":    "National line",
    "other:Regionální":           "Regional",
    "other:PronajatéRegDráhy":    "Leased regional",
    "other:SoukroméRegDráhy":     "Private regional",
    "other:Vlečky":               "Siding/industrial",
}

PREDEFINED_VEHICLES: dict[str, dict] = {
    "EDITA (diesel railcar)": dict(
        traction="DIESEL", mass=22_000, length=15,
        power=152, aux_power=20, accel=0.5, decel=0.8, efficiency=0.30),
    "EDITA+Btax": dict(
        traction="DIESEL", mass=42_000, length=30,
        power=152, aux_power=25, accel=0.4, decel=0.8, efficiency=0.30),
    "RegioNova (Class 814)": dict(
        traction="DIESEL", mass=80_000, length=44,
        power=485, aux_power=20, accel=0.5, decel=0.8, efficiency=0.32),
    "Stadler RS1 (Class 840)": dict(
        traction="DIESEL", mass=50_000, length=25.5,
        power=514, aux_power=25, accel=0.8, decel=0.9, efficiency=0.38),
    "CityElefant (Class 471)": dict(
        traction="ELECTRIC", mass=155_000, length=79,
        power=2_000, aux_power=80, accel=0.8, decel=0.9, efficiency=0.85),
}


# ─────────────────────────────────────────────────────────────────────────────
#  DYPOD RAILML PARSER
# ─────────────────────────────────────────────────────────────────────────────
class DYPODParser:
    """
    Parses a DYPOD railML 3.2 export and exposes:
      op_info   : dict  op_id  -> {name, lat, lon, types, n_tracks, connected_lines}
      seg_props : dict  ne_id  -> {speed_normal, speed_reverse, grad_normal,
                                    grad_reverse, electrification, recuperation,
                                    length_m, braking_distance, line_category,
                                    n_tracks, mode_of_op, gps_start, gps_end}
      graph     : dict  op_id  -> [{to, ne_id, length_m, forward}, ...]
      op_by_name: dict  name   -> [op_id, ...]
    """

    def __init__(self, xml_path: str):
        t0 = time.time()
        self._root = self._strip_ns(ET.parse(xml_path).getroot())
        self._esys: dict[str, str] = {}
        self._sub_to_seg: dict[str, str] = {}
        self.op_info:    dict[str, dict] = {}
        self.seg_props:  dict[str, dict] = {}
        self.graph:      dict[str, list] = {}
        self.op_by_name: dict[str, list] = {}

        self._parse_esys()
        self._parse_sub_map()
        self._parse_ops()
        self._parse_segs()
        self._build_graph()
        self._build_name_idx()
        self.parse_time = round(time.time() - t0, 2)

    # ── XML ────────────────────────────────────────────────────────────────
    @staticmethod
    def _strip_ns(root: ET.Element) -> ET.Element:
        for el in root.iter():
            if "}" in el.tag:
                el.tag = el.tag.split("}", 1)[1]
        return root

    # ── Electrification systems ────────────────────────────────────────────
    def _parse_esys(self):
        for es in self._root.iter("electrificationSystem"):
            eid = es.get("id", "")
            v, f = es.get("voltage", "0"), es.get("frequency", "0")
            self._esys[eid] = "NONE" if v == "0" else f"{int(float(v)):,}V / {f}Hz"

    # ── Track-sub → segment map ────────────────────────────────────────────
    def _parse_sub_map(self):
        for ne in self._root.iter("netElement"):
            ne_id = ne.get("id", "")
            ecu = ne.find("elementCollectionUnordered")
            if ecu is not None:
                for ep in ecu.findall("elementPart"):
                    ref = ep.get("ref", "")
                    if ref:
                        self._sub_to_seg[ref] = ne_id

    # ── Operational points ─────────────────────────────────────────────────
    def _parse_ops(self):
        for op in self._root.iter("operationalPoint"):
            op_id = op.get("id", "")
            name_el = op.find("name")
            name = name_el.get("name", op_id) if name_el is not None else op_id

            lat = lon = None
            for sl in op.findall("spotLocation"):
                geo = sl.find("geometricCoordinate")
                if geo is not None and geo.get("positioningSystemRef", "") == "gps01":
                    lat = float(geo.get("x", 0))
                    lon = float(geo.get("y", 0))
                    break

            optypes  = [x.get("operationalType", "") for x in op.iter("opOperation")]
            connected = [x.get("ref", "") for x in op.findall("connectedToLine")]
            eq = op.find("opEquipment")
            n_tracks = int(eq.get("numberOfStationTracks", 1)) if eq is not None else 1
            des = op.find("designator")
            sr70 = des.get("entry", "") if des is not None else ""

            self.op_info[op_id] = dict(
                name=name, lat=lat, lon=lon, types=optypes,
                n_tracks=n_tracks, connected_lines=connected, sr70=sr70,
            )

    # ── Segment properties ─────────────────────────────────────────────────
    def _parse_segs(self):
        # ── speed ──────────────────────────────────────────────────────────
        speed_map: dict[str, dict] = {}
        for ss in self._root.iter("speedSection"):
            ss_id = ss.get("id", "")
            maxsp = float(ss.get("maxSpeed", 80))
            ane = ss.find(".//associatedNetElement")
            if ane is None:
                continue
            seg = self._sub_to_seg.get(ane.get("netElementRef", ""),
                                        ane.get("netElementRef", ""))
            d = "reverse" if "_reverse" in ss_id else "normal"
            speed_map.setdefault(seg, {})[d] = maxsp

        # ── gradient ───────────────────────────────────────────────────────
        grad_map: dict[str, dict] = {}
        for gc in self._root.iter("gradientCurve"):
            grad = float(gc.get("gradient", 0))
            loc  = gc.find("linearLocation")
            if loc is None:
                continue
            app_dir = loc.get("applicationDirection", "normal")
            ane = loc.find("associatedNetElement")
            if ane is None:
                continue
            ne_ref = ane.get("netElementRef", "")
            grad_map.setdefault(ne_ref, {})[app_dir] = grad

        # ── electrification ────────────────────────────────────────────────
        elec_map: dict[str, str] = {}
        for sec in self._root.iter("electrificationSection"):
            ane    = sec.find(".//associatedNetElement")
            es_ref = sec.find("electrificationSystemRef")
            if ane is None or es_ref is None:
                continue
            seg   = self._sub_to_seg.get(ane.get("netElementRef", ""),
                                          ane.get("netElementRef", ""))
            label = self._esys.get(es_ref.get("ref", ""), "UNKNOWN")
            # keep any non-NONE label (multiple sections may touch same segment)
            elec_map[seg] = label if label != "NONE" else elec_map.get(seg, "NONE")

        # ── track physical length ──────────────────────────────────────────
        len_map: dict[str, float] = {}
        for tr in self._root.iter("track"):
            ane = tr.find(".//associatedNetElement")
            lel = tr.find("length")
            if ane is None or lel is None:
                continue
            seg = self._sub_to_seg.get(ane.get("netElementRef", ""),
                                        ane.get("netElementRef", ""))
            ltype = lel.get("type", "")
            val   = float(lel.get("value", 0))
            if ltype == "physical" or seg not in len_map:
                len_map[seg] = val

        # ── GPS endpoints from netElements ─────────────────────────────────
        gps_map: dict[str, tuple] = {}
        for ne in self._root.iter("netElement"):
            ne_id = ne.get("id", "")
            for aps in ne.findall("associatedPositioningSystem"):
                if aps.get("positioningSystemRef", "") != "gps01":
                    continue
                ics = aps.findall("intrinsicCoordinate")
                if len(ics) < 2:
                    continue
                coords = []
                for ic in ics:
                    geo = ic.find("geometricCoordinate")
                    if geo is not None:
                        coords.append((float(geo.get("x", 0)),
                                       float(geo.get("y", 0))))
                if len(coords) >= 2:
                    gps_map[ne_id] = (coords[0], coords[-1])

        # ── Assemble from line elements ────────────────────────────────────
        for line in self._root.iter("line"):
            ane = line.find(".//associatedNetElement")
            if ane is None:
                continue
            ne_ref = ane.get("netElementRef", "")
            lel = line.find("length")
            length_m = (float(lel.get("value", 0)) if lel is not None
                        else len_map.get(ne_ref, 0.0))
            sp  = speed_map.get(ne_ref, {})
            gr  = grad_map.get(ne_ref, {})
            electr = elec_map.get(ne_ref, "NONE")
            gps    = gps_map.get(ne_ref, (None, None))
            lp     = line.find("linePerformance")
            ll     = line.find("lineLayout")
            lo     = line.find("lineOperation")

            self.seg_props[ne_ref] = dict(
                line_id       = line.get("id", ""),
                length_m      = length_m,
                speed_normal  = sp.get("normal",  sp.get("reverse", 80.0)),
                speed_reverse = sp.get("reverse", sp.get("normal",  80.0)),
                grad_normal   = gr.get("normal",  0.0),
                grad_reverse  = gr.get("reverse", 0.0),
                electrification = electr,
                recuperation  = 0 if electr == "NONE" else 1,
                braking_dist  = int(lp.get("signalledBrakingDistance", 0)) if lp is not None else 0,
                line_category = LINE_CATEGORY_LABELS.get(
                                    line.get("lineCategory", ""), line.get("lineCategory", "")),
                n_tracks      = ll.get("numberOfTracks", "single") if ll is not None else "single",
                mode_of_op    = lo.get("modeOfOperation", "") if lo is not None else "",
                gps_start     = gps[0],
                gps_end       = gps[1],
            )

    # ── Graph ──────────────────────────────────────────────────────────────
    def _build_graph(self):
        for line in self._root.iter("line"):
            b_el = line.find("beginsInOP")
            e_el = line.find("endsInOP")
            if b_el is None or e_el is None:
                continue
            b_op   = b_el.get("ref", "")
            e_op   = e_el.get("ref", "")
            ane    = line.find(".//associatedNetElement")
            ne_ref = ane.get("netElementRef", "") if ane is not None else ""
            length_m = self.seg_props.get(ne_ref, {}).get("length_m", 0.0)
            self.graph.setdefault(b_op, []).append(
                dict(to=e_op, ne_id=ne_ref, length_m=length_m, forward=True))
            self.graph.setdefault(e_op, []).append(
                dict(to=b_op, ne_id=ne_ref, length_m=length_m, forward=False))

    # ── Name index ─────────────────────────────────────────────────────────
    def _build_name_idx(self):
        for op_id, info in self.op_info.items():
            self.op_by_name.setdefault(info["name"], []).append(op_id)

    # ── Dijkstra path search ───────────────────────────────────────────────
    def dijkstra(self, start: str, end: str) -> tuple[float | None, list[dict]]:
        """Returns (total_m, edge_path) or (None, []) if unreachable."""
        if start not in self.graph:
            return None, []
        tb  = itertools.count()
        pq  = [(0.0, next(tb), start, [])]
        vis: set = set()
        while pq:
            cost, _, node, path = heapq.heappop(pq)
            if node in vis:
                continue
            vis.add(node)
            if node == end:
                return cost, path
            for edge in self.graph.get(node, []):
                nxt = edge["to"]
                if nxt not in vis:
                    heapq.heappush(pq, (
                        cost + edge["length_m"], next(tb), nxt,
                        path + [dict(from_op=node, to_op=nxt,
                                     ne_id=edge["ne_id"],
                                     length_m=edge["length_m"],
                                     forward=edge["forward"])],
                    ))
        return None, []

    # ── Profile builder ────────────────────────────────────────────────────
    def build_profile(self, start_op: str, end_op: str) -> pd.DataFrame:
        """
        Stitches the meso-level segments into a direction-aware track profile.

        Returns DataFrame columns:
          cum_km         – cumulative distance from start [km]
          station_name   – waypoint display name
          stop_type      – 'X' mandatory passenger stop, 'R' request, '' pass-through
          speed_kmh      – max speed in travel direction [km/h]
          gradient_perm  – grade in travel direction [‰] (positive = uphill)
          electrification – system label or 'NONE'
          recuperation   – 1 if electrified segment, else 0
          length_m       – physical segment length [m]
          braking_dist   – signalled braking distance [m]
          line_category  – line category label
          n_tracks       – 'single' / 'double' / 'mixed' / 'multiple'
          lat, lon       – WGS-84 GPS of waypoint
          op_types       – comma-separated operational type(s)
        """
        _, path = self.dijkstra(start_op, end_op)
        if not path:
            return pd.DataFrame()

        rows: list[dict] = []
        cum_m = 0.0

        def _seg_dir(seg: dict, forward: bool) -> dict:
            """Return direction-resolved segment properties."""
            return dict(
                speed_kmh      = seg.get("speed_normal"  if forward else "speed_reverse", 80.0),
                gradient_perm  = seg.get("grad_normal"   if forward else "grad_reverse",  0.0),
                electrification= seg.get("electrification", "NONE"),
                recuperation   = seg.get("recuperation", 0),
                length_m       = seg.get("length_m", 0.0),
                braking_dist   = seg.get("braking_dist", 0),
                line_category  = seg.get("line_category", ""),
                n_tracks       = seg.get("n_tracks", "single"),
            )

        def _op_stop_type(op_id: str) -> str:
            types = self.op_info.get(op_id, {}).get("types", [])
            return "X" if any(t in PASSENGER_STOP_TYPES for t in types) else ""

        for step in path:
            from_op = step["from_op"]
            ne_id   = step["ne_id"]
            fwd     = step["forward"]
            seg     = self.seg_props.get(ne_id, {})
            sd      = _seg_dir(seg, fwd)
            info    = self.op_info.get(from_op, {})

            rows.append(dict(
                cum_km        = cum_m / 1000.0,
                station_name  = info.get("name", from_op),
                stop_type     = _op_stop_type(from_op),
                lat           = info.get("lat"),
                lon           = info.get("lon"),
                op_types      = ", ".join(info.get("types", [])),
                **sd,
            ))
            cum_m += seg.get("length_m", 0.0)

        # Final destination
        last_fwd = path[-1]["forward"]
        last_seg = self.seg_props.get(path[-1]["ne_id"], {})
        last_sd  = _seg_dir(last_seg, last_fwd)
        info_end = self.op_info.get(end_op, {})
        rows.append(dict(
            cum_km        = cum_m / 1000.0,
            station_name  = info_end.get("name", end_op),
            stop_type     = "X",   # destination is always a stop
            lat           = info_end.get("lat"),
            lon           = info_end.get("lon"),
            op_types      = ", ".join(info_end.get("types", [])),
            length_m      = 0.0,
            **{k: v for k, v in last_sd.items() if k != "length_m"},
        ))

        df = pd.DataFrame(rows)
        df["cum_km"] = df["cum_km"].round(4)
        return df

    # ── Station search ─────────────────────────────────────────────────────
    def search_stations(self, query: str,
                        max_results: int = 60) -> list[tuple[str, str]]:
        q = query.strip().lower()
        if not q:
            return []
        out = []
        for op_id, info in self.op_info.items():
            if not any(t in PASSENGER_STOP_TYPES for t in info.get("types", [])):
                continue
            if q in info["name"].lower():
                out.append((op_id, info["name"]))
        return sorted(out, key=lambda x: x[1])[:max_results]

    def get_stations_list(self) -> list[tuple[str, str]]:
        """All passenger OPs sorted by name: [(op_id, name), ...]"""
        out = []
        for op_id, info in self.op_info.items():
            if any(t in PASSENGER_STOP_TYPES for t in info.get("types", [])):
                out.append((op_id, info["name"]))
        return sorted(out, key=lambda x: x[1])


# ─────────────────────────────────────────────────────────────────────────────
#  TRACK PROFILE  (physics layer)
# ─────────────────────────────────────────────────────────────────────────────
class TrackProfile:
    """Wraps a profile DataFrame and provides per-position physics queries."""

    def __init__(self, df: pd.DataFrame):
        self.df       = df.copy().reset_index(drop=True)
        self.segments = self._build_segments()
        self.stations = self._extract_stations()
        self.total_km = float(df["cum_km"].max())

    def _build_segments(self) -> list[dict]:
        segs = []
        for i in range(len(self.df) - 1):
            r = self.df.iloc[i]
            segs.append(dict(
                km_start      = float(r["cum_km"]),
                km_end        = float(self.df.iloc[i+1]["cum_km"]),
                v_limit       = float(r["speed_kmh"]) / 3.6,
                grad          = float(r["gradient_perm"]) / 1000.0,
                electrification = str(r.get("electrification", "NONE")),
                recuperation  = int(r.get("recuperation", 0)),
            ))
        return segs

    def _extract_stations(self) -> list[dict]:
        out = []
        for _, row in self.df.iterrows():
            name  = str(row.get("station_name", "")).strip()
            stype = str(row.get("stop_type", "")).strip().upper()
            if name and stype in ("X", "R"):
                out.append(dict(name=name, km=float(row["cum_km"]), type=stype))
        return out

    def seg_at(self, km: float) -> dict:
        for seg in self.segments:
            if seg["km_start"] <= km <= seg["km_end"] + 1e-6:
                return seg
        return self.segments[-1] if self.segments else {}

    def v_limit_span(self, front_km: float, rear_km: float) -> float:
        """Minimum speed limit [m/s] over the train body."""
        lo, hi = min(front_km, rear_km), max(front_km, rear_km)
        limits = [s["v_limit"] for s in self.segments
                  if lo <= s["km_end"] + 1e-6 and hi >= s["km_start"] - 1e-6]
        return min(limits) if limits else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  TRAIN SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
class TrainSimulator:
    """
    1-D energy simulation:
      - Davis resistance    F_res = A + B·v + C·v²
      - Gradient force      F_grad = m·g·slope
      - Traction power limit
      - Recuperative braking (electric only, when segment allows)
    """

    def __init__(self, mass_kg, length_m, max_power_kw, aux_power_kw,
                 max_accel, max_decel, traction_type, efficiency):
        self.mass_kg     = mass_kg
        self.eff_mass    = mass_kg * 1.08      # rotational inertia factor
        self.A, self.B, self.C = 1500.0, 30.0, 4.0
        self.traction    = traction_type
        self.trac_eff    = efficiency if traction_type == "ELECTRIC" else 0.85
        self.regen_eff   = 0.75 if traction_type == "ELECTRIC" else 0.0
        self.aux_w       = aux_power_kw  * 1_000.0
        self.max_w       = max_power_kw  * 1_000.0
        self.max_accel   = max_accel
        self.max_decel   = max_decel
        self.length_m    = length_m

    def _resistance(self, v: float) -> float:
        return self.A + self.B * v + self.C * v * v

    def run(self, track: TrackProfile,
            stop_mode: str = "all",
            stop_prob: float = 1.0,
            dwell_time: float = 30.0,
            record_history: bool = False,
            ) -> tuple[dict | None, list[str], dict]:
        """
        Simulate the full trip (0 → track.total_km).
        Returns (history | None, stops_served_names, stats_dict).
        stats keys: net_kwh, gross_kwh, regen_kwh, journey_time_s.
        """
        total_m = track.total_km * 1000.0
        g = 9.8186

        # Decide which intermediate stations to stop at
        stops_km:    list[float] = []
        stops_names: list[str]   = []
        for st in track.stations:
            km = st["km"]
            if km <= 1e-3 or km >= track.total_km - 1e-3:
                continue
            will_stop = False
            if st["type"] == "X":
                will_stop = True
            elif st["type"] == "R":
                will_stop = (stop_mode == "all") or (
                    stop_mode == "random" and random.random() <= stop_prob)
            if will_stop:
                stops_km.append(km)
                stops_names.append(st["name"])

        dt     = 1.0
        v      = 0.0
        dist_m = 0.0
        e_j    = 0.0    # gross energy [J]
        r_j    = 0.0    # recuperated [J]
        t_s    = 0.0    # journey time [s]

        hist = ({k: [] for k in
                 ("time_s", "km", "v_kmh", "v_limit_kmh",
                  "gross_kwh", "regen_kwh", "net_kwh")}
                if record_history else None)

        while dist_m < total_m:
            km       = dist_m / 1000.0
            rear_km  = max(0.0, km - self.length_m / 1000.0)
            seg      = track.seg_at(km)
            v_lim    = track.v_limit_span(km, rear_km)
            slope    = seg.get("grad", 0.0)
            electr   = seg.get("electrification", "NONE")
            recup_ok = seg.get("recuperation", 0) == 1

            if self.traction == "ELECTRIC" and electr == "NONE":
                raise RuntimeError(
                    f"Electric vehicle enters non-electrified segment at km {km:.3f}. "
                    "Choose a diesel vehicle or select an electrified route.")

            f_grad    = self.mass_kg * g * slope
            eff_decel = max(0.05, self.max_decel + g * slope)

            # Braking look-ahead for upcoming stops
            max_safe = v_lim
            for s_km in stops_km:
                if s_km < km - 0.01:
                    continue
                d2s = (s_km - km) * 1000.0
                max_safe = min(max_safe, math.sqrt(max(0.0, 2.0 * eff_decel * d2s)))

            # ── Record history snapshot ────────────────────────────────────
            if hist is not None:
                hist["time_s"].append(t_s)
                hist["km"].append(km)
                hist["v_kmh"].append(v * 3.6)
                hist["v_limit_kmh"].append(v_lim * 3.6)
                hist["gross_kwh"].append(e_j / 3_600_000.0)
                hist["regen_kwh"].append(r_j / 3_600_000.0)
                hist["net_kwh"].append((e_j - r_j) / 3_600_000.0)

            # ── Physics step ───────────────────────────────────────────────
            v_eff  = max(v, 0.5)
            mech_p = 0.0
            reg_p  = 0.0

            if v > max_safe + 1e-4:                        # BRAKING
                f_res     = self._resistance(v)
                nat_dec   = (f_res + f_grad) / self.eff_mass
                req_dec   = (v - max_safe) / dt
                brake     = max(0.0, min(self.max_decel, req_dec - nat_dec))
                if recup_ok:
                    reg_p = self.eff_mass * brake * v * self.regen_eff
                v = max(0.0, v - (brake + nat_dec) * dt)

            elif v < min(v_lim, max_safe) - 1e-4:          # TRACTION
                f_res    = self._resistance(v)
                des_f    = f_res + self.eff_mass * self.max_accel + f_grad
                act_f    = max(0.0, min(des_f, self.max_w / v_eff))
                v        = max(0.0, min(v + (act_f - f_res - f_grad) /
                                         self.eff_mass * dt, v_lim, max_safe))
                mech_p   = max(0.0, act_f * v)

            else:                                           # CRUISE
                f_res    = self._resistance(v)
                des_f    = f_res + f_grad
                act_f    = max(0.0, min(des_f, self.max_w / v_eff))
                v        = max(0.0, v + (act_f - f_res - f_grad) /
                                         self.eff_mass * dt)
                mech_p   = max(0.0, act_f * v)

            e_j    += (mech_p / self.trac_eff + self.aux_w) * dt
            r_j    += reg_p * dt
            dist_m += v * dt
            t_s    += dt

            # ── Dwell at stop ──────────────────────────────────────────────
            if v < 0.5 and any(abs(km - s) <= 0.05 for s in stops_km):
                v = 0.0
                if hist is not None:
                    for pass_no in range(2):
                        vl = track.v_limit_span(km, km) * 3.6
                        hist["time_s"].append(t_s)
                        hist["km"].append(km)
                        hist["v_kmh"].append(0.0)
                        hist["v_limit_kmh"].append(vl)
                        hist["gross_kwh"].append(e_j / 3_600_000.0)
                        hist["regen_kwh"].append(r_j / 3_600_000.0)
                        hist["net_kwh"].append((e_j - r_j) / 3_600_000.0)
                        if pass_no == 0:
                            e_j += self.aux_w * dwell_time
                            t_s += dwell_time
                else:
                    e_j += self.aux_w * dwell_time
                    t_s += dwell_time
                stops_km = [s for s in stops_km if abs(s - km) > 0.05]

        stats = dict(
            gross_kwh     = e_j / 3_600_000.0,
            regen_kwh     = r_j / 3_600_000.0,
            net_kwh       = (e_j - r_j) / 3_600_000.0,
            journey_time_s= t_s,
        )
        return hist, stops_names, stats


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_dur(sec: float) -> str:
    sec = int(sec)
    h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"

def to_unit(stats: dict, traction: str, density: float, eff: float) -> float:
    return stats["net_kwh"] / (density * eff) if traction == "DIESEL" else stats["net_kwh"]

def load_xml_from_upload(uploaded) -> str | None:
    """Extract railML xml from an uploaded file (xml/railml or zip). Returns path."""
    ext = uploaded.name.lower().rsplit(".", 1)[-1]
    tmp = tempfile.mkdtemp(prefix="dypod_")
    if ext == "zip":
        zp = os.path.join(tmp, uploaded.name)
        with open(zp, "wb") as f:
            f.write(uploaded.read())
        with zipfile.ZipFile(zp) as zf:
            xmls = [n for n in zf.namelist()
                    if n.lower().endswith((".xml", ".railml"))]
            if not xmls:
                return None
            zf.extract(xmls[0], tmp)
            return os.path.join(tmp, xmls[0])
    else:
        path = os.path.join(tmp, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.read())
        return path


# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DYPOD Track Profile & Fleet Simulator",
    page_icon="🚆",
    layout="wide",
)

# Session state keys
for _k in ("parser", "profile_df", "rep_result", "mc_result",
           "xml_path", "start_op", "end_op"):
    if _k not in st.session_state:
        st.session_state[_k] = None


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🚆 DYPOD Simulator")

    # ── File loading ─────────────────────────────────────────────────────────
    st.header("📂 1. Load railML")
    uploaded = st.file_uploader(
        "Upload railML file or zip",
        type=["xml", "railml", "zip"],
        help="DYPOD railML 3.2 export from Správa železnic.",
    )

    xml_path: str | None = None
    if uploaded is not None:
        xml_path = load_xml_from_upload(uploaded)
        if xml_path and xml_path != st.session_state.xml_path:
            st.session_state.xml_path     = xml_path
            st.session_state.parser       = None
            st.session_state.profile_df   = None
            st.session_state.rep_result   = None
            st.session_state.mc_result    = None
    else:
        # fallback: local files
        locals_ = sorted(
            glob.glob("*.xml") + glob.glob("*.railml") + glob.glob("*.zip") +
            glob.glob("/tmp/*.xml") + glob.glob("/tmp/*.railml")
        )
        if locals_:
            sel = st.selectbox("Or pick a local file", locals_)
            if sel != st.session_state.xml_path:
                st.session_state.xml_path   = sel
                st.session_state.parser     = None
                st.session_state.profile_df = None
                st.session_state.rep_result = None
                st.session_state.mc_result  = None
            xml_path = sel

    # Parse (cached at resource level)
    @st.cache_resource(show_spinner="Parsing railML — please wait ~5 s…")
    def _load(path: str) -> DYPODParser:
        if path.lower().endswith(".zip"):
            extracted = load_xml_from_upload(
                type("F", (), {"name": path,
                               "read": lambda self: open(path, "rb").read()})()
            )
            return DYPODParser(extracted or path)
        return DYPODParser(path)

    parser: DYPODParser | None = None
    if xml_path:
        try:
            parser = _load(xml_path)
            st.success(
                f"✅ Parsed in {parser.parse_time} s\n\n"
                f"**{len(parser.op_info):,}** OPs · "
                f"**{len(parser.seg_props):,}** segments"
            )
        except Exception as exc:
            st.error(f"Parse error: {exc}")

    if parser is None:
        st.info("Upload or select a DYPOD railML file above.")
        st.stop()

    # ── Route selection ──────────────────────────────────────────────────────
    st.header("🗺️ 2. Route")

    q_a = st.text_input("Search departure station", placeholder="e.g. Praha")
    q_b = st.text_input("Search arrival station",   placeholder="e.g. Brno")

    res_a = parser.search_stations(q_a) if q_a.strip() else []
    res_b = parser.search_stations(q_b) if q_b.strip() else []

    op_start = op_end = None

    if res_a:
        lbl_a = st.selectbox("Departure", [r[1] for r in res_a], key="sel_a")
        op_start = next((r[0] for r in res_a if r[1] == lbl_a), None)
    elif q_a.strip():
        st.warning("No match — try a different name.")

    if res_b:
        lbl_b = st.selectbox("Arrival", [r[1] for r in res_b], key="sel_b")
        op_end = next((r[0] for r in res_b if r[1] == lbl_b), None)
    elif q_b.strip():
        st.warning("No match — try a different name.")

    btn_profile = st.button("🗺️ Build Track Profile",
                             use_container_width=True, type="primary",
                             disabled=not (op_start and op_end and op_start != op_end))

    # ── Vehicle ──────────────────────────────────────────────────────────────
    st.header("🚃 3. Vehicle")
    veh = st.selectbox("Preset", ["Custom"] + list(PREDEFINED_VEHICLES.keys()))
    if veh == "Custom":
        traction    = st.selectbox("Traction", ["DIESEL", "ELECTRIC"])
        mass        = st.number_input("Mass (kg)",          value=100_000, step=5_000)
        length      = st.number_input("Length (m)",         value=40,      step=5)
        power       = st.number_input("Max Power (kW)",     value=500,     step=50)
        aux_power   = st.number_input("Aux Power (kW)",     value=40,      step=5)
        accel       = st.slider("Max Accel (m/s²)",  0.2, 1.5, 0.6, 0.05)
        decel       = st.slider("Max Brake (m/s²)",  0.4, 1.5, 0.9, 0.05)
        efficiency  = st.slider("Efficiency (%)",    15,  95,  35)  / 100.0
        diesel_dens = st.number_input("Diesel density (kWh/L)", value=10.0, step=0.1)
    else:
        p = PREDEFINED_VEHICLES[veh]
        traction, mass, length = p["traction"], p["mass"], p["length"]
        power, aux_power        = p["power"], p["aux_power"]
        accel, decel, efficiency = p["accel"], p["decel"], p["efficiency"]
        diesel_dens = 10.0

    unit_lbl = "L (diesel)" if traction == "DIESEL" else "kWh"

    # ── Simulation settings ──────────────────────────────────────────────────
    st.header("⚙️ 4. Simulation")
    dwell       = st.number_input("Station dwell (s)",  value=30, step=5)
    stop_mode   = st.selectbox("Stop mode (representative run)",
                                ["all", "random", "none"])
    stop_prob   = st.slider("Request-stop probability", 0.0, 1.0, 0.5, 0.05,
                             disabled=(stop_mode != "random"))

    st.header("🎲 5. Monte Carlo")
    mc_n     = st.number_input("Runs per probability (N)",
                                min_value=20, max_value=500, value=100, step=10)
    mc_probs = st.multiselect(
        "Probabilities to sweep",
        options=[1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
        default=[1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
    )

    c1, c2 = st.columns(2)
    btn_run = c1.button("▶️ Run",        use_container_width=True,
                         disabled=st.session_state.profile_df is None)
    btn_mc  = c2.button("🎲 Monte Carlo", use_container_width=True,
                         disabled=st.session_state.profile_df is None)


# ──────────────────────────────────────────────────────────────────────────────
#  ACTIONS
# ──────────────────────────────────────────────────────────────────────────────

# Build profile
if btn_profile and op_start and op_end:
    with st.spinner("Finding shortest path and stitching profile…"):
        df = parser.build_profile(op_start, op_end)
    if df.empty:
        st.error("No path found between the selected stations. "
                 "They may not be connected in this export.")
    else:
        st.session_state.profile_df = df
        st.session_state.start_op   = op_start
        st.session_state.end_op     = op_end
        st.session_state.rep_result = None
        st.session_state.mc_result  = None

# Representative run
if btn_run and st.session_state.profile_df is not None:
    with st.spinner("Running physics simulation…"):
        try:
            track = TrackProfile(st.session_state.profile_df)
            sim   = TrainSimulator(mass, length, power, aux_power,
                                   accel, decel, traction, efficiency)
            hist, snames, stats = sim.run(
                track, stop_mode=stop_mode, stop_prob=stop_prob,
                dwell_time=dwell, record_history=True)
            st.session_state.rep_result = dict(
                hist=hist, stop_names=snames, stats=stats,
                traction=traction, diesel_dens=diesel_dens, efficiency=efficiency)
        except RuntimeError as e:
            st.error(str(e))

# Monte Carlo
if btn_mc and st.session_state.profile_df is not None and mc_probs:
    with st.spinner(f"Monte Carlo: {int(mc_n)} runs × {len(mc_probs)} probabilities…"):
        try:
            track = TrackProfile(st.session_state.profile_df)
            sim   = TrainSimulator(mass, length, power, aux_power,
                                   accel, decel, traction, efficiency)
            rows_mc = []
            for p_val in sorted(mc_probs, reverse=True):
                e_list, t_list = [], []
                sm = ("all" if p_val == 1.0 else "none" if p_val == 0.0
                      else "random")
                for _ in range(int(mc_n)):
                    _, _, st_ = sim.run(track, stop_mode=sm,
                                         stop_prob=p_val, dwell_time=dwell)
                    e_list.append(to_unit(st_, traction, diesel_dens, efficiency))
                    t_list.append(st_["journey_time_s"])
                rows_mc.append(dict(
                    prob_label = f"{int(p_val*100)}%",
                    p_num      = p_val,
                    mean_e     = float(np.mean(e_list)),
                    std_e      = float(np.std(e_list)),
                    min_e      = float(np.min(e_list)),
                    max_e      = float(np.max(e_list)),
                    mean_t     = float(np.mean(t_list)),
                    min_t      = float(np.min(t_list)),
                    max_t      = float(np.max(t_list)),
                ))
            mc_df = pd.DataFrame(rows_mc)
            worst = mc_df["mean_e"].max()
            mc_df["savings"] = (worst - mc_df["mean_e"]).clip(lower=0)
            mc_df["unit"] = unit_lbl
            st.session_state.mc_result = mc_df
        except RuntimeError as e:
            st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA  (tabs)
# ══════════════════════════════════════════════════════════════════════════════
tab_prof, tab_run, tab_mc_tab = st.tabs([
    "🗺️ Track Profile", "▶️ Representative Run", "🎲 Monte Carlo",
])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 – TRACK PROFILE
# ─────────────────────────────────────────────────────────────────────────────
with tab_prof:
    df = st.session_state.profile_df
    if df is None:
        st.info("👈 Search start/end stations in the sidebar and click **Build Track Profile**.")
        st.stop()

    total_km  = float(df["cum_km"].max())
    n_stops   = int((df["stop_type"] == "X").sum())
    max_spd   = float(df["speed_kmh"].max())
    max_grad  = float(df["gradient_perm"].abs().max())

    # Electrified km
    elec_km = sum(
        (df.iloc[i+1]["cum_km"] - df.iloc[i]["cum_km"])
        for i in range(len(df)-1)
        if df.iloc[i]["electrification"] != "NONE"
    )
    elec_pct = 100.0 * elec_km / total_km if total_km > 0 else 0

    # Route header
    st.subheader(
        f"📍 {df['station_name'].iloc[0]}  →  {df['station_name'].iloc[-1]}"
    )

    # KPI row
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Route length",       f"{total_km:.1f} km")
    k2.metric("Waypoints",          f"{len(df)}")
    k3.metric("Passenger stops",    f"{n_stops}")
    k4.metric("Max speed",          f"{max_spd:.0f} km/h")
    k5.metric("Max gradient",       f"{max_grad:.0f} ‰")
    k6.metric("Electrified",        f"{elec_km:.0f} km ({elec_pct:.0f}%)")

    # ── Speed + gradient + electrification chart ──────────────────────────
    stops_df = df[df["stop_type"] == "X"]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Speed profile [km/h]",
                         "Gradient [‰]  (+ uphill, − downhill)",
                         "Electrification"),
        row_heights=[0.50, 0.30, 0.20],
        vertical_spacing=0.06,
    )

    # Speed fill
    fig.add_trace(go.Scatter(
        x=df["cum_km"], y=df["speed_kmh"],
        mode="lines", name="Speed limit",
        line=dict(color="#2563EB", width=2.5),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.10)",
    ), row=1, col=1)

    # Station diamond markers
    fig.add_trace(go.Scatter(
        x=stops_df["cum_km"], y=stops_df["speed_kmh"],
        mode="markers+text",
        marker=dict(symbol="diamond", size=8, color="#EA580C",
                    line=dict(color="white", width=1)),
        text=stops_df["station_name"],
        textposition="top center",
        textfont=dict(size=7.5),
        name="Passenger stops",
    ), row=1, col=1)

    # Gradient bars coloured by direction
    bar_w = max(total_km / len(df) * 0.9, 0.01)
    pos_g = df["gradient_perm"].clip(lower=0)
    neg_g = df["gradient_perm"].clip(upper=0)
    fig.add_trace(go.Bar(
        x=df["cum_km"], y=pos_g, name="Uphill",
        marker_color="#EF4444", width=bar_w,
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=df["cum_km"], y=neg_g, name="Downhill",
        marker_color="#3B82F6", width=bar_w,
    ), row=2, col=1)

    # Electrification colour band
    elec_unique = df["electrification"].unique().tolist()
    elec_colors = {
        "NONE":          "#9CA3AF",
        "25,000V / 50Hz":"#16A34A",
        "3,000V / 0Hz":  "#7C3AED",
        "1,500V / 0Hz":  "#0891B2",
        "15,000V / 16.7Hz": "#D97706",
    }
    # Build colour for each row
    e_colors = [elec_colors.get(e, "#6B7280") for e in df["electrification"]]
    fig.add_trace(go.Bar(
        x=df["cum_km"], y=[1] * len(df),
        marker_color=e_colors,
        name="Electrification",
        hovertext=df["electrification"],
        hovertemplate="%{hovertext}<extra></extra>",
        width=bar_w, showlegend=False,
    ), row=3, col=1)

    # Vertical stop lines on speed row
    for _, srow in stops_df.iterrows():
        fig.add_vline(x=srow["cum_km"], line_width=0.8,
                      line_dash="dot", line_color="#6B7280", row=1, col=1)

    fig.update_xaxes(title_text="Distance from departure [km]", row=3, col=1)
    fig.update_yaxes(title_text="Speed [km/h]", row=1, col=1)
    fig.update_yaxes(title_text="Gradient [‰]",  row=2, col=1)
    fig.update_yaxes(showticklabels=False,        row=3, col=1)
    fig.update_layout(
        height=720, barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Electrification legend row ────────────────────────────────────────
    for esys, color in elec_colors.items():
        if esys in elec_unique:
            km_e = sum(
                (df.iloc[i+1]["cum_km"] - df.iloc[i]["cum_km"])
                for i in range(len(df)-1)
                if df.iloc[i]["electrification"] == esys
            )
            st.markdown(
                f'<span style="background:{color};color:white;'
                f'padding:2px 8px;border-radius:4px;font-size:0.82em">'
                f'{esys}</span>  **{km_e:.1f} km** '
                f'{"(recuperation ✅)" if esys != "NONE" else "(no recuperation ❌)"}   ',
                unsafe_allow_html=True,
            )

    st.write("")  # spacer

    # ── Route map ─────────────────────────────────────────────────────────
    map_df = df.dropna(subset=["lat", "lon"])
    if not map_df.empty:
        with st.expander("🗺️ Route map", expanded=True):
            # Colour line by electrification
            fig_m = go.Figure()
            for esys in df["electrification"].unique():
                seg_df = map_df[map_df["electrification"] == esys]
                color  = elec_colors.get(esys, "#6B7280")
                fig_m.add_trace(go.Scattermapbox(
                    lat=seg_df["lat"], lon=seg_df["lon"],
                    mode="lines+markers",
                    marker=dict(size=4, color=color),
                    line=dict(color=color, width=4),
                    name=esys,
                    hovertemplate=f"{esys}<extra></extra>",
                ))
            # Passenger stop markers
            smap = map_df[map_df["stop_type"] == "X"]
            fig_m.add_trace(go.Scattermapbox(
                lat=smap["lat"], lon=smap["lon"],
                mode="markers+text",
                marker=dict(size=11, color="#EA580C",
                            symbol="circle"),
                text=smap["station_name"],
                textposition="top right",
                textfont=dict(size=9),
                name="Passenger stops",
            ))
            fig_m.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(
                    center=dict(lat=float(map_df["lat"].mean()),
                                lon=float(map_df["lon"].mean())),
                    zoom=7,
                ),
                height=500, margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig_m, use_container_width=True)

    # ── Profile table ─────────────────────────────────────────────────────
    with st.expander("📋 Detailed profile table"):
        show_cols = ["cum_km", "station_name", "stop_type", "speed_kmh",
                     "gradient_perm", "electrification", "recuperation",
                     "length_m", "braking_dist", "line_category",
                     "n_tracks", "op_types"]
        show_cols = [c for c in show_cols if c in df.columns]
        rename = dict(
            cum_km="km", station_name="Waypoint", stop_type="Stop",
            speed_kmh="Speed [km/h]", gradient_perm="Gradient [‰]",
            electrification="Electrification", recuperation="Recup.",
            length_m="Length [m]", braking_dist="Braking dist [m]",
            line_category="Line category", n_tracks="Tracks", op_types="Type",
        )
        st.dataframe(df[show_cols].rename(columns=rename),
                     use_container_width=True, hide_index=True)
        csv = df[show_cols].to_csv(index=False).encode()
        st.download_button("⬇️ Download profile CSV", csv,
                           "track_profile.csv", "text/csv")

    # ── Segment summary by electrification ───────────────────────────────
    with st.expander("⚡ Electrification breakdown"):
        rows_e = []
        for i in range(len(df)-1):
            rows_e.append(dict(
                System   = df.iloc[i]["electrification"],
                km       = round(df.iloc[i+1]["cum_km"] - df.iloc[i]["cum_km"], 3),
                Recup    = "Yes" if df.iloc[i]["recuperation"] else "No",
            ))
        edf = (pd.DataFrame(rows_e)
               .groupby(["System", "Recup"])["km"].sum()
               .reset_index()
               .sort_values("km", ascending=False))
        edf["% of route"] = (edf["km"] / total_km * 100).round(1)
        st.dataframe(edf, use_container_width=True, hide_index=True)

    # ── Gradient statistics ────────────────────────────────────────────────
    with st.expander("📐 Gradient statistics"):
        g_vals = df["gradient_perm"].dropna()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max uphill",   f"{g_vals.max():.1f} ‰")
        c2.metric("Max downhill", f"{g_vals.min():.1f} ‰")
        c3.metric("Mean",         f"{g_vals.mean():.1f} ‰")
        c4.metric("Std dev",      f"{g_vals.std():.1f} ‰")

        fig_g = px.histogram(
            g_vals, nbins=30,
            labels={"value": "Gradient [‰]", "count": "Segments"},
            title="Gradient distribution across segments",
            color_discrete_sequence=["#2563EB"],
        )
        st.plotly_chart(fig_g, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 – REPRESENTATIVE RUN
# ─────────────────────────────────────────────────────────────────────────────
with tab_run:
    rep = st.session_state.rep_result
    if rep is None:
        st.info("👈 Build a profile first, then click **▶️ Run**.")
        st.stop()

    hist  = rep["hist"]
    stats = rep["stats"]
    snames= rep["stop_names"]
    _tr   = rep["traction"]
    _dd   = rep["diesel_dens"]
    _eff  = rep["efficiency"]
    consumed = to_unit(stats, _tr, _dd, _eff)

    df_p = st.session_state.profile_df
    tot_km = float(df_p["cum_km"].max()) if df_p is not None else 0

    st.subheader("Representative Run Results")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Journey time",     fmt_dur(stats["journey_time_s"]))
    c2.metric(f"Net ({unit_lbl})", f"{consumed:.2f}")
    c3.metric("Gross (kWh)",      f"{stats['gross_kwh']:.2f}")
    c4.metric("Recuperated (kWh)",f"{stats['regen_kwh']:.2f}")
    c5.metric("Avg speed",
              f"{tot_km / (stats['journey_time_s'] / 3600):.1f} km/h"
              if stats["journey_time_s"] > 0 else "—")
    c6.metric("Stops served",     str(len(snames)))

    if snames:
        st.markdown("**Stops served:** " + "  →  ".join(snames))

    if hist:
        # Speed + energy subplots
        fig2 = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("Speed [km/h]", "Cumulative energy [kWh]"),
            vertical_spacing=0.10,
        )
        fig2.add_trace(go.Scatter(
            x=hist["km"], y=hist["v_limit_kmh"],
            name="Speed limit", line=dict(color="#EF4444", dash="dot", width=1.5),
        ), row=1, col=1)
        fig2.add_trace(go.Scatter(
            x=hist["km"], y=hist["v_kmh"],
            name="Actual speed", line=dict(color="#2563EB", width=2),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
        ), row=1, col=1)
        fig2.add_trace(go.Scatter(
            x=hist["km"], y=hist["gross_kwh"],
            name="Gross energy", line=dict(color="#F59E0B", width=2),
        ), row=2, col=1)
        fig2.add_trace(go.Scatter(
            x=hist["km"], y=hist["regen_kwh"],
            name="Recuperated", line=dict(color="#16A34A", width=2, dash="dash"),
        ), row=2, col=1)
        fig2.add_trace(go.Scatter(
            x=hist["km"], y=hist["net_kwh"],
            name="Net energy", line=dict(color="#7C3AED", width=2),
        ), row=2, col=1)

        # Station markers on speed plot
        if df_p is not None:
            stops_r = df_p[df_p["stop_type"] == "X"]
            for _, sr in stops_r.iterrows():
                fig2.add_vline(x=sr["cum_km"], line_width=0.8,
                               line_dash="dot", line_color="#6B7280", row=1, col=1)

        fig2.update_xaxes(title_text="Distance from departure [km]", row=2, col=1)
        fig2.update_yaxes(title_text="Speed [km/h]",    row=1, col=1)
        fig2.update_yaxes(title_text="Energy [kWh]",    row=2, col=1)
        fig2.update_layout(
            height=560,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Speed histogram
        with st.expander("📊 Speed distribution"):
            fig_sv = px.histogram(
                hist["v_kmh"], nbins=40,
                labels={"value": "Speed [km/h]", "count": "Seconds at speed"},
                title="Time spent at each speed",
                color_discrete_sequence=["#2563EB"],
            )
            st.plotly_chart(fig_sv, use_container_width=True)

    # Specific metrics
    if stats["gross_kwh"] > 0 and tot_km > 0:
        regen_pct  = stats["regen_kwh"] / stats["gross_kwh"] * 100
        spec_e     = stats["net_kwh"]   / tot_km
        st.markdown("---")
        ca, cb, cc = st.columns(3)
        ca.metric("Recuperation share",  f"{regen_pct:.1f}%")
        cb.metric("Specific net energy", f"{spec_e:.3f} kWh/km")
        if _tr == "DIESEL":
            spec_fuel = consumed / tot_km * 1000
            cc.metric("Specific fuel",   f"{spec_fuel:.1f} mL/km")
        else:
            cc.metric("Net kWh / km", f"{spec_e:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 – MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────
with tab_mc_tab:
    mc_df = st.session_state.mc_result
    if mc_df is None:
        st.info("👈 Build a profile first, then click **🎲 Monte Carlo**.")
        st.stop()

    ul = mc_df["unit"].iloc[0]
    st.subheader(f"Monte Carlo Analysis  —  N = {int(mc_n)} runs per probability")

    # ── Summary table ────────────────────────────────────────────────────
    display_mc = mc_df.rename(columns=dict(
        prob_label=f"Stop probability",
        mean_e=f"Mean ({ul})", std_e=f"Std ({ul})",
        min_e=f"Min ({ul})",  max_e=f"Max ({ul})",
        mean_t="Mean time", min_t="Fastest", max_t="Slowest",
        savings=f"Savings vs all-stops ({ul})",
    ))
    display_mc["Mean time"] = display_mc["Mean time"].apply(
        lambda x: fmt_dur(x) if isinstance(x, (int, float)) else x)
    display_mc["Fastest"]   = display_mc["Fastest"].apply(
        lambda x: fmt_dur(x) if isinstance(x, (int, float)) else x)
    display_mc["Slowest"]   = display_mc["Slowest"].apply(
        lambda x: fmt_dur(x) if isinstance(x, (int, float)) else x)
    drop_cols = ["p_num", "unit"]
    st.dataframe(
        display_mc.drop(columns=[c for c in drop_cols if c in display_mc.columns]),
        use_container_width=True, hide_index=True,
    )
    csv_mc = mc_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download MC CSV", csv_mc, "mc_results.csv", "text/csv")

    # ── Charts ───────────────────────────────────────────────────────────
    ca, cb = st.columns(2)

    with ca:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=mc_df["prob_label"],
            y=mc_df["savings"],
            marker=dict(
                color=mc_df["savings"],
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title=ul),
            ),
            text=mc_df["savings"].round(2),
            texttemplate="%{text}",
            textposition="outside",
        ))
        fig_bar.update_layout(
            title=f"Energy savings vs. all-stops baseline [{ul}]",
            xaxis_title="Request-stop probability",
            yaxis_title=f"Savings [{ul}]",
            height=420,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with cb:
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(
            x=mc_df["prob_label"],
            y=mc_df["mean_e"],
            error_y=dict(type="data", array=mc_df["std_e"].tolist(), visible=True,
                         color="#2563EB", thickness=2, width=8),
            mode="markers+lines",
            marker=dict(size=11, color="#2563EB"),
            line=dict(color="#2563EB", width=2),
            name="Mean ± 1 Std",
        ))
        # Min-max band
        fig_err.add_trace(go.Scatter(
            x=pd.concat([mc_df["prob_label"], mc_df["prob_label"].iloc[::-1]]),
            y=pd.concat([mc_df["max_e"], mc_df["min_e"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(37,99,235,0.10)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Min–Max range",
            showlegend=True,
        ))
        fig_err.update_layout(
            title=f"Mean ± std and min/max range [{ul}]",
            xaxis_title="Request-stop probability",
            yaxis_title=f"Energy [{ul}]",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_err, use_container_width=True)

    # ── Journey time spread ───────────────────────────────────────────────
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(
        x=mc_df["prob_label"], y=mc_df["mean_t"] / 60,
        mode="markers+lines", name="Mean journey time",
        marker=dict(size=10, color="#EA580C"),
        line=dict(color="#EA580C", width=2),
    ))
    fig_t.add_trace(go.Scatter(
        x=pd.concat([mc_df["prob_label"], mc_df["prob_label"].iloc[::-1]]),
        y=pd.concat([mc_df["max_t"] / 60, mc_df["min_t"].iloc[::-1] / 60]),
        fill="toself", fillcolor="rgba(234,88,12,0.10)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Min–Max range",
    ))
    fig_t.update_layout(
        title="Journey time vs. stopping policy",
        xaxis_title="Request-stop probability",
        yaxis_title="Journey time [min]",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_t, use_container_width=True)
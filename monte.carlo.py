"""
DYPOD Track Profile Builder & Fleet Energy Simulator
=====================================================
• Parses DYPOD railML 3.2 (Správa železnic / OLTIS Group)
• Builds direction-aware stitched track profiles
  - Speed: min across all sub-track speedSections per direction
  - Gradient: normal/reverse from gradientCurve applicationDirection
  - Electrification & recuperation per segment
• Live station search with dropdown suggestions
• Interactive profile editor with via-waypoints + table overrides
• Smart electrified-route search (penalty-based rerouting)
• Electric coasting through short non-electrified gaps (configurable)
• Physics simulation (Davis resistance, traction, recuperation)
• Monte Carlo stochastic stop-probability analysis

Run:  streamlit run dypod_simulator.py
"""
from __future__ import annotations
import glob, heapq, itertools, math, os, random, tempfile, time, zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── colour palette ────────────────────────────────────────────────────────────
C = dict(
    primary="#2563EB", secondary="#7C3AED", accent="#EA580C",
    green="#16A34A",   yellow="#CA8A04",    red="#DC2626",
    grey="#6B7280",    light="#F1F5F9",     dark="#1E293B",
    bg_blue="rgba(37,99,235,0.08)", bg_orange="rgba(234,88,12,0.08)",
)
ELEC_COLORS = {
    "NONE":           "#9CA3AF",
    "25,000V/50Hz":   "#16A34A",
    "3,000V/0Hz":     "#7C3AED",
    "1,500V/0Hz":     "#0891B2",
    "15,000V/16.7Hz": "#D97706",
}
PASSENGER_TYPES = {"station", "stoppingPoint"}
MANDATORY_TYPES = {"station"}
REQUEST_TYPES   = {"stoppingPoint"}

PREDEFINED_VEHICLES = {
    "EDITA (diesel railcar)":  dict(traction="DIESEL",  mass=22_000,  length=15,   power=152,   aux_power=20, accel=0.5, decel=0.8, efficiency=0.30),
    "EDITA+Btax":              dict(traction="DIESEL",  mass=42_000,  length=30,   power=152,   aux_power=25, accel=0.4, decel=0.8, efficiency=0.30),
    "RegioNova (Class 814)":   dict(traction="DIESEL",  mass=80_000,  length=44,   power=485,   aux_power=20, accel=0.5, decel=0.8, efficiency=0.32),
    "Stadler RS1 (Class 840)": dict(traction="DIESEL",  mass=50_000,  length=25.5, power=514,   aux_power=25, accel=0.8, decel=0.9, efficiency=0.38),
    "CityElefant (Class 471)": dict(traction="ELECTRIC",mass=155_000, length=79,   power=2_000, aux_power=80, accel=0.8, decel=0.9, efficiency=0.85),
}

# ─────────────────────────────────────────────────────────────────────────────
#  DYPOD railML PARSER
# ─────────────────────────────────────────────────────────────────────────────
class DYPODParser:
    """
    Parses a DYPOD railML 3.2 export.

    Speed handling
    --------------
    Each meso-level segment (line element) can contain multiple track
    sub-elements (neCZ…_1, _1a, _2 …).  Each sub-element has its own
    speedSection pair (_normal / _reverse).  We take the MINIMUM speed
    across all sub-tracks per direction — the most restrictive value that
    any train on the segment must honour.
    For the 82 segments with no speedSection we fall back to
    linePerformance.maxSpeed (if available) then to 30 km/h (siding/depot).

    Gradient handling
    -----------------
    gradientCurve.applicationDirection ∈ {normal, reverse} maps directly
    to travel direction (beginsInOP→endsInOP = normal, reverse = opposite).
    """

    def __init__(self, xml_path: str):
        t0 = time.time()
        self._root = self._strip_ns(ET.parse(xml_path).getroot())
        self._esys:      dict[str, str]              = {}
        self._sub2seg:   dict[str, str]              = {}
        self._seg_speeds:dict[str, dict[str, list]]  = defaultdict(lambda: {"normal": [], "reverse": []})
        self._line_max:  dict[str, int]              = {}   # ne_ref -> linePerformance maxSpeed
        self.op_info:    dict[str, dict]             = {}
        self.seg_props:  dict[str, dict]             = {}
        self.graph:      dict[str, list]             = {}
        self.op_by_name: dict[str, list]             = {}
        # sorted list of (name, op_id) for search
        self.station_list: list[tuple[str, str]]     = []

        self._parse_esys()
        self._parse_sub2seg()
        self._parse_speed_sections()   # builds _seg_speeds
        self._parse_ops()
        self._parse_segs()
        self._build_graph()
        self._build_indexes()
        self.parse_time = round(time.time() - t0, 2)

    # ── XML ───────────────────────────────────────────────────────────────────
    @staticmethod
    def _strip_ns(root: ET.Element) -> ET.Element:
        for el in root.iter():
            if "}" in el.tag:
                el.tag = el.tag.split("}", 1)[1]
        return root

    # ── Electrification systems ───────────────────────────────────────────────
    def _parse_esys(self):
        for es in self._root.iter("electrificationSystem"):
            v, f = es.get("voltage", "0"), es.get("frequency", "0")
            self._esys[es.get("id", "")] = (
                "NONE" if v == "0" else f"{int(float(v)):,}V/{f}Hz"
            )

    # ── Track-sub → segment map ────────────────────────────────────────────────
    def _parse_sub2seg(self):
        for ne in self._root.iter("netElement"):
            ne_id = ne.get("id", "")
            ecu = ne.find("elementCollectionUnordered")
            if ecu is not None:
                for ep in ecu.findall("elementPart"):
                    ref = ep.get("ref", "")
                    if ref:
                        self._sub2seg[ref] = ne_id

    # ── Speed sections (min-of-all-sub-tracks strategy) ────────────────────────
    def _parse_speed_sections(self):
        for ss in self._root.iter("speedSection"):
            ss_id = ss.get("id", "")
            maxsp = float(ss.get("maxSpeed", 0))
            if maxsp <= 0:
                continue
            ane = ss.find(".//associatedNetElement")
            if ane is None:
                continue
            sub_ne = ane.get("netElementRef", "")
            seg_ne = self._sub2seg.get(sub_ne, sub_ne)
            direction = "reverse" if "_reverse" in ss_id else "normal"
            self._seg_speeds[seg_ne][direction].append(maxsp)

    def _speed_for(self, ne_ref: str, direction: str) -> float:
        """
        Returns the direction-specific speed limit for a segment.
        Takes the minimum across all sub-track speed sections.
        Falls back to linePerformance maxSpeed, then to 30 km/h.
        """
        vals = self._seg_speeds.get(ne_ref, {}).get(direction, [])
        if vals:
            return float(min(vals))
        # Try opposite direction (some segments only have one direction)
        other = "reverse" if direction == "normal" else "normal"
        vals_other = self._seg_speeds.get(ne_ref, {}).get(other, [])
        if vals_other:
            return float(min(vals_other))
        # Fall back to linePerformance
        lp = self._line_max.get(ne_ref, 0)
        return float(lp) if lp > 0 else 30.0   # 30 km/h for siding/depot connections

    # ── Operational points ─────────────────────────────────────────────────────
    def _parse_ops(self):
        for op in self._root.iter("operationalPoint"):
            oid = op.get("id", "")
            nm = op.find("name")
            name = nm.get("name", oid) if nm is not None else oid
            lat = lon = None
            for sl in op.findall("spotLocation"):
                geo = sl.find("geometricCoordinate")
                if geo is not None and geo.get("positioningSystemRef", "") == "gps01":
                    lat, lon = float(geo.get("x", 0)), float(geo.get("y", 0))
                    break
            types    = [x.get("operationalType", "") for x in op.iter("opOperation")]
            conn     = [x.get("ref", "") for x in op.findall("connectedToLine")]
            eq       = op.find("opEquipment")
            n_tracks = int(eq.get("numberOfStationTracks", 1)) if eq is not None else 1
            des      = op.find("designator")
            sr70     = des.get("entry", "") if des is not None else ""
            self.op_info[oid] = dict(
                name=name, lat=lat, lon=lon, types=types,
                n_tracks=n_tracks, connected_lines=conn, sr70=sr70,
            )

    # ── Segment properties ─────────────────────────────────────────────────────
    def _parse_segs(self):
        # Gradient
        grad_map: dict[str, dict] = {}
        for gc in self._root.iter("gradientCurve"):
            grad = float(gc.get("gradient", 0))
            loc  = gc.find("linearLocation")
            if loc is None:
                continue
            ane = loc.find("associatedNetElement")
            if ane is None:
                continue
            ne_ref = ane.get("netElementRef", "")
            grad_map.setdefault(ne_ref, {})[loc.get("applicationDirection", "normal")] = grad

        # Electrification
        elec_map: dict[str, str] = {}
        for sec in self._root.iter("electrificationSection"):
            ane = sec.find(".//associatedNetElement")
            esr = sec.find("electrificationSystemRef")
            if ane is None or esr is None:
                continue
            seg   = self._sub2seg.get(ane.get("netElementRef", ""), ane.get("netElementRef", ""))
            label = self._esys.get(esr.get("ref", ""), "UNKNOWN")
            elec_map[seg] = label if label != "NONE" else elec_map.get(seg, "NONE")

        # GPS endpoints from netElement
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
                        coords.append((float(geo.get("x", 0)), float(geo.get("y", 0))))
                if len(coords) >= 2:
                    gps_map[ne_id] = (coords[0], coords[-1])
                    break

        # Physical length from track element
        len_map: dict[str, float] = {}
        for tr in self._root.iter("track"):
            ane = tr.find(".//associatedNetElement")
            lel = tr.find("length")
            if ane is None or lel is None:
                continue
            seg   = self._sub2seg.get(ane.get("netElementRef", ""), ane.get("netElementRef", ""))
            ltype = lel.get("type", "")
            val   = float(lel.get("value", 0))
            if ltype == "physical" or seg not in len_map:
                len_map[seg] = val

        # Assemble per line element
        for line in self._root.iter("line"):
            ane = line.find(".//associatedNetElement")
            if ane is None:
                continue
            ne_ref = ane.get("netElementRef", "")
            lel    = line.find("length")
            length_m = float(lel.get("value", 0)) if lel is not None else len_map.get(ne_ref, 0.0)
            lp  = line.find("linePerformance")
            lo  = line.find("lineOperation")
            ll  = line.find("lineLayout")
            # Store linePerformance maxSpeed for fallback
            lp_spd = int(lp.get("maxSpeed", 0)) if lp is not None else 0
            self._line_max[ne_ref] = lp_spd

            gr     = grad_map.get(ne_ref, {})
            electr = elec_map.get(ne_ref, "NONE")
            gps    = gps_map.get(ne_ref, (None, None))

            self.seg_props[ne_ref] = dict(
                line_id        = line.get("id", ""),
                length_m       = length_m,
                speed_normal   = self._speed_for(ne_ref, "normal"),
                speed_reverse  = self._speed_for(ne_ref, "reverse"),
                grad_normal    = gr.get("normal",  0.0),
                grad_reverse   = gr.get("reverse", 0.0),
                electrification= electr,
                recuperation   = 0 if electr == "NONE" else 1,
                braking_dist   = int(lp.get("signalledBrakingDistance", 0)) if lp is not None else 0,
                n_tracks       = ll.get("numberOfTracks", "single") if ll is not None else "single",
                mode_of_op     = lo.get("modeOfOperation", "") if lo is not None else "",
                gps_start      = gps[0],
                gps_end        = gps[1],
            )

    # ── Graph ──────────────────────────────────────────────────────────────────
    def _build_graph(self):
        for line in self._root.iter("line"):
            b_el = line.find("beginsInOP")
            e_el = line.find("endsInOP")
            ane  = line.find(".//associatedNetElement")
            if b_el is None or e_el is None or ane is None:
                continue
            b_op, e_op = b_el.get("ref", ""), e_el.get("ref", "")
            ne_ref     = ane.get("netElementRef", "")
            props      = self.seg_props.get(ne_ref, {})
            length_m   = props.get("length_m", 0.0)
            electr     = props.get("electrification", "NONE")
            self.graph.setdefault(b_op, []).append(
                dict(to=e_op, ne_id=ne_ref, length_m=length_m, forward=True,  electr=electr))
            self.graph.setdefault(e_op, []).append(
                dict(to=b_op, ne_id=ne_ref, length_m=length_m, forward=False, electr=electr))

    # ── Search indexes ─────────────────────────────────────────────────────────
    def _build_indexes(self):
        for oid, info in self.op_info.items():
            self.op_by_name.setdefault(info["name"], []).append(oid)
            if any(t in PASSENGER_TYPES for t in info.get("types", [])):
                self.station_list.append((info["name"], oid))
        self.station_list.sort(key=lambda x: x[0])

    # ── Dijkstra ───────────────────────────────────────────────────────────────
    def dijkstra(self, start: str, end: str,
                 unelec_penalty_m: float = 0.0) -> tuple[float | None, list]:
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
                if nxt in vis:
                    continue
                pen = unelec_penalty_m if edge["electr"] == "NONE" else 0.0
                heapq.heappush(pq, (
                    cost + edge["length_m"] + pen,
                    next(tb), nxt,
                    path + [dict(from_op=node, **edge)],
                ))
        return None, []

    # ── Electrification analysis ───────────────────────────────────────────────
    def analyse_electrification(self, start_op: str, end_op: str) -> dict:
        _, p_normal = self.dijkstra(start_op, end_op, 0)
        if not p_normal:
            return dict(normal_km=0, normal_ue_km=0, gateway_km=0,
                        gateway_op=None, penalised_km=0, penalised_ue_km=0,
                        detour_km=0, elec_saving_km=0, has_alternative=False)

        def _ue_km(path):
            return sum(self.seg_props.get(s["ne_id"], {}).get("length_m", 0)
                       for s in path if s["electr"] == "NONE") / 1000

        def _tot_km(path):
            return sum(self.seg_props.get(s["ne_id"], {}).get("length_m", 0)
                       for s in path) / 1000

        normal_ue  = _ue_km(p_normal)
        normal_tot = _tot_km(p_normal)

        gateway_km = 0.0
        gateway_op = None
        for s in p_normal:
            if s["electr"] != "NONE":
                gateway_op = s["from_op"]
                break
            gateway_km += self.seg_props.get(s["ne_id"], {}).get("length_m", 0) / 1000

        _, p_pen   = self.dijkstra(start_op, end_op, 100_000)
        pen_ue     = _ue_km(p_pen)
        pen_tot    = _tot_km(p_pen)
        elec_saving = normal_ue - pen_ue
        detour      = pen_tot - normal_tot
        has_alt     = elec_saving >= 1.0 and detour <= max(elec_saving * 10, 30)

        return dict(
            normal_km=round(normal_tot, 1),    normal_ue_km=round(normal_ue, 1),
            gateway_km=round(gateway_km, 1),   gateway_op=gateway_op,
            penalised_km=round(pen_tot, 1),    penalised_ue_km=round(pen_ue, 1),
            detour_km=round(detour, 1),        elec_saving_km=round(elec_saving, 1),
            has_alternative=has_alt,
        )

    # ── Profile builder ────────────────────────────────────────────────────────
    def build_profile(self, start_op: str, end_op: str,
                      via_ops: list[str] | None = None,
                      unelec_penalty_m: float = 0.0) -> pd.DataFrame | None:
        waypoints = [start_op] + (via_ops or []) + [end_op]
        all_steps: list[dict] = []
        for i in range(len(waypoints) - 1):
            _, path = self.dijkstra(waypoints[i], waypoints[i + 1], unelec_penalty_m)
            if not path:
                return None
            all_steps.extend(path)
        if not all_steps:
            return None

        def _stop_type(oid: str) -> str:
            types = self.op_info.get(oid, {}).get("types", [])
            if any(t in MANDATORY_TYPES for t in types): return "X"
            if any(t in REQUEST_TYPES   for t in types): return "R"
            return ""

        def _sd(seg: dict, fwd: bool) -> dict:
            return dict(
                speed_kmh       = seg.get("speed_normal" if fwd else "speed_reverse", 30.0),
                gradient_perm   = seg.get("grad_normal"  if fwd else "grad_reverse",  0.0),
                electrification = seg.get("electrification", "NONE"),
                recuperation    = seg.get("recuperation", 0),
                length_m        = seg.get("length_m", 0.0),
                braking_dist    = seg.get("braking_dist", 0),
                n_tracks        = seg.get("n_tracks", "single"),
            )

        # Pre-compute consecutive NONE-gap ahead of each step
        ue_gap_ahead: list[float] = []
        for i, step in enumerate(all_steps):
            if self.seg_props.get(step["ne_id"], {}).get("electrification", "NONE") != "NONE":
                ue_gap_ahead.append(0.0)
                continue
            gap = 0.0
            j = i
            while j < len(all_steps):
                s = self.seg_props.get(all_steps[j]["ne_id"], {})
                if s.get("electrification", "NONE") != "NONE":
                    break
                gap += s.get("length_m", 0.0)
                j += 1
            ue_gap_ahead.append(gap)

        rows: list[dict] = []
        cum_m = 0.0
        for idx, step in enumerate(all_steps):
            from_op = step["from_op"]
            seg     = self.seg_props.get(step["ne_id"], {})
            sd      = _sd(seg, step["forward"])
            info    = self.op_info.get(from_op, {})
            rows.append(dict(
                cum_km        = cum_m / 1000.0,
                op_id         = from_op,
                station_name  = info.get("name", from_op),
                stop_type     = _stop_type(from_op),
                lat           = info.get("lat"),
                lon           = info.get("lon"),
                op_types      = ", ".join(info.get("types", [])),
                ue_gap_m      = ue_gap_ahead[idx],
                **sd,
            ))
            cum_m += seg.get("length_m", 0.0)

        # Final destination row
        last_fwd = all_steps[-1]["forward"]
        last_seg = self.seg_props.get(all_steps[-1]["ne_id"], {})
        last_sd  = _sd(last_seg, last_fwd)
        info_end = self.op_info.get(end_op, {})
        rows.append(dict(
            cum_km        = cum_m / 1000.0,
            op_id         = end_op,
            station_name  = info_end.get("name", end_op),
            stop_type     = "X",
            lat           = info_end.get("lat"),
            lon           = info_end.get("lon"),
            op_types      = ", ".join(info_end.get("types", [])),
            ue_gap_m      = 0.0,
            length_m      = 0.0,
            **{k: v for k, v in last_sd.items() if k != "length_m"},
        ))

        df = pd.DataFrame(rows)
        df["cum_km"] = df["cum_km"].round(4)
        return df

    # ── Station search ─────────────────────────────────────────────────────────
    def search_stations(self, query: str, max_results: int = 80) -> list[tuple[str, str]]:
        """Return [(op_id, name)] matching query. Empty query → [] ."""
        q = query.strip().lower()
        if not q:
            return []
        return [(oid, nm) for nm, oid in self.station_list
                if q in nm.lower()][:max_results]

    def op_name(self, op_id: str) -> str:
        return self.op_info.get(op_id, {}).get("name", op_id)


# ─────────────────────────────────────────────────────────────────────────────
#  TRACK PROFILE  (physics wrapper)
# ─────────────────────────────────────────────────────────────────────────────
class TrackProfile:
    def __init__(self, df: pd.DataFrame):
        self.df       = df.copy().reset_index(drop=True)
        if self.df.empty:
            self.total_km = 0.0
            self.segments = []
            self.stations = []
        else:
            self.total_km = float(self.df["cum_km"].max())
            self.segments = self._build_segs()
            self.stations = self._build_stations()

    def _build_segs(self) -> list[dict]:
        segs = []
        for i in range(len(self.df) - 1):
            r = self.df.iloc[i]
            segs.append(dict(
                km_start        = float(r["cum_km"]),
                km_end          = float(self.df.iloc[i + 1]["cum_km"]),
                v_limit         = float(r["speed_kmh"]) / 3.6,
                grad            = float(r["gradient_perm"]) / 1000.0,
                electrification = str(r.get("electrification", "NONE")),
                recuperation    = int(r.get("recuperation", 0)),
                ue_gap_m        = float(r.get("ue_gap_m", 0.0)),
            ))
        return segs

    def _build_stations(self) -> list[dict]:
        return [
            dict(name=str(r.get("station_name", "")), km=float(r["cum_km"]),
                 type=str(r.get("stop_type", "")).upper())
            for _, r in self.df.iterrows()
            if str(r.get("stop_type", "")).upper() in ("X", "R")
            and str(r.get("station_name", "")).strip()
        ]

    def seg_at(self, km: float) -> dict:
        for s in self.segments:
            if s["km_start"] <= km <= s["km_end"] + 1e-6:
                return s
        return self.segments[-1] if self.segments else {}

    def v_limit_span(self, front_km: float, rear_km: float) -> float:
        if not self.segments:
            return 0.0
        lo, hi = min(front_km, rear_km), max(front_km, rear_km)
        lims = [s["v_limit"] for s in self.segments
                if lo <= s["km_end"] + 1e-6 and hi >= s["km_start"] - 1e-6]
        return min(lims) if lims else self.segments[-1]["v_limit"]

    @property
    def n_mandatory(self) -> int:
        return sum(1 for s in self.stations if s["type"] == "X")

    @property
    def n_request(self) -> int:
        return sum(1 for s in self.stations if s["type"] == "R")


# ─────────────────────────────────────────────────────────────────────────────
#  TRAIN SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
class TrainSimulator:
    """
    1-D physics simulation.

    Electric vehicle coasting
    --------------------------
    When an electric vehicle enters a non-electrified segment whose
    consecutive gap length ≤ coast_threshold_m the vehicle coasts:
      • traction force = 0
      • no regenerative braking
      • auxiliary power (HVAC, lighting) still consumed
      • natural deceleration from Davis resistance + gradient
    Gaps longer than the threshold raise RuntimeError.
    """

    def __init__(self, mass_kg, length_m, max_power_kw, aux_power_kw,
                 max_accel, max_decel, traction_type, efficiency,
                 coast_threshold_m: float = 500.0):
        self.mass_kg  = mass_kg
        self.eff_mass = mass_kg * 1.08          # rotational inertia factor
        self.A, self.B, self.C = 1500.0, 30.0, 4.0   # Davis coefficients
        self.traction   = traction_type
        self.trac_eff   = efficiency if traction_type == "ELECTRIC" else 0.85
        self.regen_eff  = 0.75 if traction_type == "ELECTRIC" else 0.0
        self.aux_w      = aux_power_kw * 1_000.0
        self.max_w      = max_power_kw * 1_000.0
        self.max_accel  = max_accel
        self.max_decel  = max_decel
        self.length_m   = length_m
        self.coast_threshold_m = coast_threshold_m

    def _res(self, v: float) -> float:
        return self.A + self.B * v + self.C * v * v

    def run(self, track: TrackProfile, stop_mode: str = "all",
            stop_prob: float = 1.0, dwell: float = 30.0,
            record: bool = False):
        total_m = track.total_km * 1000.0
        g = 9.8186

        # Build list of stops to serve
        stops_km:   list[float] = []
        stop_names: list[str]   = []
        for st in track.stations:
            km = st["km"]
            if km <= 1e-3: # Always skip the starting point
                continue
            if km >= track.total_km - 1e-3:
                continue # We manually append the destination later
            will = (st["type"] == "X") or (
                st["type"] == "R" and (
                    stop_mode == "all" or
                    (stop_mode == "random" and random.random() <= stop_prob)
                )
            )
            if will:
                stops_km.append(km)
                stop_names.append(st["name"])

        # Train MUST ALWAYS stop at the final destination
        if track.total_km > 0.0:
            stops_km.append(track.total_km)
            stop_names.append("Destination")

        v = dist = e_j = r_j = t_s = 0.0
        hist = {k: [] for k in ("time_s", "km", "v_kmh", "v_limit_kmh",
                                 "gross_kwh", "regen_kwh", "net_kwh")} if record else None

        # Watchdog limit to prevent infinite loops on impossible climbs
        max_iters = int(total_m / 0.1) + 36000
        iters = 0

        while dist < total_m:
            iters += 1
            if iters > max_iters:
                raise RuntimeError(f"Simulation stalled indefinitely at km {km:.2f} (Speed dropped to 0 and cannot overcome resistance/gradient).")

            km       = dist / 1000.0
            rear_km  = max(0.0, km - self.length_m / 1000.0)
            seg      = track.seg_at(km)
            v_lim    = track.v_limit_span(km, rear_km)
            slope    = seg.get("grad", 0.0)
            electr   = seg.get("electrification", "NONE")
            recup_ok = seg.get("recuperation", 0) == 1

            # ── Electrification guard ──────────────────────────────────────
            coasting = False
            if self.traction == "ELECTRIC" and electr == "NONE":
                gap_m = seg.get("ue_gap_m", 0.0)
                if gap_m <= self.coast_threshold_m:
                    coasting = True          # coast on inertia
                else:
                    raise RuntimeError(
                        f"⚡ Electric vehicle on non-electrified track at km {km:.2f} "
                        f"(gap {gap_m / 1000:.1f} km > coasting limit "
                        f"{self.coast_threshold_m / 1000:.1f} km). "
                        "Raise the coasting limit, enable 'Prefer electrified route', "
                        "or choose a diesel vehicle."
                    )

            f_grad    = self.mass_kg * g * slope
            eff_decel = max(0.05, self.max_decel + g * slope)

            # Braking look-ahead for the NEXT upcoming stop
            max_safe = v_lim
            next_stop = next((s for s in stops_km if s >= km - 0.01), None)
            if next_stop is not None:
                d2s = (next_stop - km) * 1000.0
                max_safe = min(max_safe, math.sqrt(max(0.0, 2.0 * eff_decel * d2s)))

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
            mech_p = reg_p = 0.0

            if coasting:
                # No traction, no regen — pure inertia decay
                f_res = self._res(v)
                v     = max(0.0, v - (f_res + f_grad) / self.eff_mass)

            elif v > max_safe + 1e-4:
                f_res    = self._res(v)
                nat_d    = (f_res + f_grad) / self.eff_mass
                brk      = max(0.0, min(self.max_decel, (v - max_safe) - nat_d))
                if recup_ok:
                    reg_p = self.eff_mass * brk * v * self.regen_eff
                v = max(0.0, v - (brk + nat_d))

            elif v < min(v_lim, max_safe) - 1e-4:
                f_res  = self._res(v)
                des_f  = f_res + self.eff_mass * self.max_accel + f_grad
                act_f  = max(0.0, min(des_f, self.max_w / v_eff))
                v      = max(0.0, min(v + (act_f - f_res - f_grad) / self.eff_mass,
                                      v_lim, max_safe))
                mech_p = max(0.0, act_f * v)

            else:
                f_res  = self._res(v)
                act_f  = max(0.0, min(f_res + f_grad, self.max_w / v_eff))
                v      = max(0.0, v + (act_f - f_res - f_grad) / self.eff_mass)
                mech_p = max(0.0, act_f * v)

            e_j    += (mech_p / self.trac_eff + self.aux_w)
            r_j    += reg_p
            dist   += v
            t_s    += 1.0

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
                            e_j += self.aux_w * dwell
                            t_s += dwell
                else:
                    e_j += self.aux_w * dwell
                    t_s += dwell

                # Check if we arrived at the final destination
                if any(abs(track.total_km - s) <= 0.05 for s in stops_km if abs(km - s) <= 0.05):
                    break

                stops_km = [s for s in stops_km if abs(s - km) > 0.05]

        return hist, stop_names, dict(
            gross_kwh      = e_j / 3_600_000.0,
            regen_kwh      = r_j / 3_600_000.0,
            net_kwh        = (e_j - r_j) / 3_600_000.0,
            journey_time_s = t_s,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_dur(s: float) -> str:
    s = int(s)
    h, r = divmod(s, 3600)
    m, sc = divmod(r, 60)
    return f"{h:02d}h {m:02d}m {sc:02d}s" if h else f"{m:02d}m {sc:02d}s"

def to_unit(stats: dict, traction: str, density: float, eff: float) -> float:
    return stats["net_kwh"] / (density * eff) if traction == "DIESEL" else stats["net_kwh"]

def elec_color(label: str) -> str:
    for k, v in ELEC_COLORS.items():
        if k in label:
            return v
    return ELEC_COLORS["NONE"]

def kpi_card(val: str, lbl: str, delta: str = "", color: str = C["primary"]) -> str:
    d = (f'<div style="font-size:.72rem;color:{color};margin-top:2px;'
         f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{delta}</div>') if delta else ""
    return (
        f'<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;'
        f'padding:12px 14px;text-align:center;height:88px;display:flex;'
        f'flex-direction:column;justify-content:center;overflow:hidden">'
        f'<div style="font-size:1.35rem;font-weight:700;color:#1E3A5F;line-height:1.2">{val}</div>'
        f'<div style="font-size:.72rem;color:#64748B;margin-top:2px">{lbl}</div>'
        f'{d}</div>'
    )

def load_xml_from_upload(uploaded) -> str | None:
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
    path = os.path.join(tmp, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.read())
    return path

def make_profile_chart(df: pd.DataFrame) -> go.Figure:
    """
    Three-panel chart: speed (stepped), gradient (bars), electrification (band).

    Speed uses line_shape='hv' (horizontal-then-vertical step) so the speed
    is drawn as a true horizontal plateau for the full length of each segment,
    with an instant vertical drop/rise exactly at the boundary km.  This
    correctly shows that 160 km/h applies from km 8.9 to km 10.9, then drops
    to 80 km/h, etc.  A plain linear interpolation would be misleading.
    """
    total_km = float(df["cum_km"].max())
    # bar width: cover each segment fully without overlap
    if len(df) > 1:
        bar_w = max(float((df["cum_km"].iloc[-1] - df["cum_km"].iloc[0])
                          / max(len(df) - 1, 1)) * 0.92, 0.005)
    else:
        bar_w = 0.5
    stops_x = df[df["stop_type"] == "X"]
    stops_r = df[df["stop_type"] == "R"]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Track Speed Limit [km/h]",
                        "Gradient [‰]  (↑ uphill  ↓ downhill)",
                        "Electrification"),
        row_heights=[0.48, 0.32, 0.20], vertical_spacing=0.055,
    )

    # ── Speed — STEPPED line (hv = hold current value until next x point) ────
    # Build hover text with station name where available
    hover_txt = []
    for _, row in df.iterrows():
        nm = str(row.get("station_name", "")).strip()
        stype = str(row.get("stop_type", "")).strip()
        tag = f"<b>{nm}</b><br>" if nm and nm != "nan" else ""
        stop_tag = {"X": " 🚉", "R": " 🛑"}.get(stype, "")
        hover_txt.append(
            f"{tag}km {row['cum_km']:.2f}{stop_tag}<br>"
            f"{row['speed_kmh']:.0f} km/h<br>"
            f"Grad: {row['gradient_perm']:.1f} ‰<br>"
            f"{row['electrification']}<extra></extra>"
        )

    fig.add_trace(go.Scatter(
        x=df["cum_km"], y=df["speed_kmh"],
        mode="lines", name="Statutory Limit",
        line=dict(color=C["primary"], width=2.5, shape="hv"),
        fill="tozeroy", fillcolor=C["bg_blue"],
        hovertemplate=hover_txt,
    ), row=1, col=1)

    # Speed-change markers — tick at each boundary where speed changes
    spd_changes = df[df["speed_kmh"].diff().abs() > 0.5].copy()
    if not spd_changes.empty:
        fig.add_trace(go.Scatter(
            x=spd_changes["cum_km"], y=spd_changes["speed_kmh"],
            mode="markers",
            marker=dict(symbol="line-ns", size=12, color=C["primary"],
                        line=dict(color=C["primary"], width=2)),
            name="Speed change",
            showlegend=False,
            hovertemplate="km %{x:.2f}<br><b>%{y:.0f} km/h</b><extra></extra>",
        ), row=1, col=1)

    # Station X markers — smart thinning to avoid label overlap
    # Show text labels for every Nth stop based on route density
    n_x = max(len(stops_x), 1)
    # Target ~20 labelled stops regardless of route length
    label_every = max(1, n_x // 20)
    labelled_x  = stops_x.iloc[::label_every]
    unlabelled_x = stops_x[~stops_x.index.isin(labelled_x.index)]

    if not labelled_x.empty:
        fig.add_trace(go.Scatter(
            x=labelled_x["cum_km"], y=labelled_x["speed_kmh"],
            mode="markers+text",
            marker=dict(symbol="diamond", size=9, color=C["accent"],
                        line=dict(color="white", width=1.5)),
            text=labelled_x["station_name"],
            textposition="top center",
            textfont=dict(size=7, color=C["dark"]),
            name="Station (X)",
            hovertemplate="<b>%{text}</b><br>km %{x:.2f}  %{y:.0f} km/h<extra></extra>",
        ), row=1, col=1)

    if not unlabelled_x.empty:
        fig.add_trace(go.Scatter(
            x=unlabelled_x["cum_km"], y=unlabelled_x["speed_kmh"],
            mode="markers",
            marker=dict(symbol="diamond", size=6, color=C["accent"],
                        line=dict(color="white", width=1)),
            showlegend=False,
            text=unlabelled_x["station_name"],
            hovertemplate="<b>%{text}</b><br>km %{x:.2f}  %{y:.0f} km/h<extra></extra>",
        ), row=1, col=1)

    # Halt R markers — marker only (text too dense on regional lines)
    if not stops_r.empty:
        fig.add_trace(go.Scatter(
            x=stops_r["cum_km"], y=stops_r["speed_kmh"],
            mode="markers",
            marker=dict(symbol="circle", size=7, color=C["yellow"],
                        line=dict(color="white", width=1)),
            name="Halt (R)",
            text=stops_r["station_name"],
            hovertemplate="<b>%{text}</b><br>km %{x:.2f}  %{y:.0f} km/h<extra></extra>",
        ), row=1, col=1)

    # ── Gradient — stepped fill ────────────────────────────────────────────────
    pos_g = df["gradient_perm"].clip(lower=0)
    neg_g = df["gradient_perm"].clip(upper=0)
    fig.add_trace(go.Scatter(
        x=df["cum_km"], y=pos_g,
        mode="lines", name="Uphill",
        line=dict(color=C["red"], width=0, shape="hv"),
        fill="tozeroy", fillcolor="rgba(220,38,38,0.55)",
        hovertemplate="km %{x:.2f}<br>+%{y:.1f} ‰<extra></extra>",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df["cum_km"], y=neg_g,
        mode="lines", name="Downhill",
        line=dict(color=C["primary"], width=0, shape="hv"),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.55)",
        hovertemplate="km %{x:.2f}<br>%{y:.1f} ‰<extra></extra>",
    ), row=2, col=1)

    # ── Electrification band — one bar per segment, exact width ───────────────
    mid_km, seg_widths, seg_colors, seg_hover = [], [], [], []
    for i in range(len(df) - 1):
        x0 = float(df["cum_km"].iloc[i])
        x1 = float(df["cum_km"].iloc[i + 1])
        mid_km.append((x0 + x1) / 2)
        seg_widths.append(max(x1 - x0, 0.001))
        seg_colors.append(elec_color(df["electrification"].iloc[i]))
        seg_hover.append(df["electrification"].iloc[i])
    if len(df) > 0:
        mid_km.append(float(df["cum_km"].iloc[-1]))
        seg_widths.append(0.001)
        seg_colors.append(elec_color(df["electrification"].iloc[-1]))
        seg_hover.append(df["electrification"].iloc[-1])

    fig.add_trace(go.Bar(
        x=mid_km, y=[1] * len(mid_km),
        width=seg_widths,
        marker_color=seg_colors,
        name="Electrification",
        hovertext=seg_hover,
        hovertemplate="%{hovertext}<br>km %{x:.2f}<extra></extra>",
        showlegend=False,
    ), row=3, col=1)

    # Vertical stop lines on speed panel — only for labelled stops
    for _, sr in labelled_x.iterrows():
        fig.add_vline(x=sr["cum_km"], line_width=0.7, line_dash="dot",
                      line_color="#CBD5E1", row=1, col=1)

    spd_min = float(df["speed_kmh"].min())
    spd_max = float(df["speed_kmh"].max())
    spd_lo  = max(0, spd_min * 0.75)   # 25% headroom below min
    spd_hi  = spd_max * 1.25           # 25% headroom above max for labels

    fig.update_xaxes(title_text="Distance from departure [km]",
                     row=3, col=1, gridcolor=C["light"])
    fig.update_yaxes(title_text="Speed [km/h]", row=1, col=1,
                     gridcolor=C["light"], range=[spd_lo, spd_hi])
    fig.update_yaxes(title_text="Gradient [‰]", row=2, col=1,
                     gridcolor=C["light"], zeroline=True, zerolinecolor="#CBD5E1")
    fig.update_yaxes(showticklabels=False, row=3, col=1, range=[0, 1.1])
    fig.update_layout(
        height=760, barmode="overlay", paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0",
                    borderwidth=1),
        margin=dict(t=70, b=10, l=60, r=20),
        font=dict(family="Inter, sans-serif", size=12),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=C["light"], showgrid=True)
    return fig

def make_route_map(df: pd.DataFrame, via_ops: list,
                   parser: DYPODParser, show_halts: bool = True) -> go.Figure:
    """Renders the route on an OpenStreetMap base."""
    map_df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    fig    = go.Figure()

    if not map_df.empty:
        runs: list[dict] = []   # {electr, lats, lons, hover}
        cur_electr  = None
        cur_lats:  list = []
        cur_lons:  list = []
        cur_hover: list = []

        for i in range(len(map_df) - 1):
            row_a = map_df.iloc[i]
            row_b = map_df.iloc[i + 1]
            seg_electr = str(row_a["electrification"])

            if seg_electr != cur_electr:
                # Save current run if non-empty
                if cur_lats:
                    runs.append(dict(electr=cur_electr,
                                     lats=cur_lats, lons=cur_lons,
                                     hover=cur_hover))
                cur_electr = seg_electr
                cur_lats   = [float(row_a["lat"])]
                cur_lons   = [float(row_a["lon"])]
                cur_hover  = [f"{seg_electr}<br>km {row_a['cum_km']:.2f}"]

            cur_lats.append(float(row_b["lat"]))
            cur_lons.append(float(row_b["lon"]))
            cur_hover.append(f"{seg_electr}<br>km {row_b['cum_km']:.2f}")

        if cur_lats:
            runs.append(dict(electr=cur_electr,
                             lats=cur_lats, lons=cur_lons, hover=cur_hover))

        seen_labels: set = set()
        for run in runs:
            show_leg = run["electr"] not in seen_labels
            seen_labels.add(run["electr"])
            col = elec_color(run["electr"])
            lw  = 5 if run["electr"] != "NONE" else 4
            fig.add_trace(go.Scattermap(
                lat=run["lats"], lon=run["lons"],
                mode="lines",
                line=dict(width=lw, color=col),
                name=run["electr"],
                showlegend=show_leg,
                hovertext=run["hover"],
                hovertemplate="%{hovertext}<extra></extra>",
            ))

    if show_halts:
        r_pts = map_df[map_df["stop_type"] == "R"]
        if not r_pts.empty:
            fig.add_trace(go.Scattermap(
                lat=r_pts["lat"].tolist(), lon=r_pts["lon"].tolist(),
                mode="markers",
                marker=dict(size=7, color=C["yellow"]),
                name="Halt (R)",
                customdata=r_pts["station_name"].values,
                hovertemplate="<b>%{customdata}</b><extra></extra>",
            ))

    x_pts = map_df[map_df["stop_type"] == "X"]
    if not x_pts.empty:
        fig.add_trace(go.Scattermap(
            lat=x_pts["lat"].tolist(), lon=x_pts["lon"].tolist(),
            mode="markers+text",
            marker=dict(size=11, color=C["accent"]),
            text=x_pts["station_name"].tolist(), textposition="top right",
            textfont=dict(size=9, color=C["dark"]),
            name="Station (X)",
            customdata=x_pts[["cum_km", "speed_kmh", "gradient_perm"]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "km %{customdata[0]:.2f}<br>"
                "%{customdata[1]:.0f} km/h<br>"
                "grad %{customdata[2]:.1f} ‰"
                "<extra></extra>"
            ),
        ))

    for vid in (via_ops or []):
        info = parser.op_info.get(vid, {})
        if info.get("lat") and info.get("lon"):
            fig.add_trace(go.Scattermap(
                lat=[info["lat"]], lon=[info["lon"]],
                mode="markers+text",
                marker=dict(size=18, color=C["yellow"]),
                text=[info["name"]], textposition="top right",
                textfont=dict(size=10, color="#92400E"),
                name=f"Via: {info['name']}", showlegend=False,
            ))

    if not map_df.empty:
        for row_lat, row_lon, row_name, tpos, col, leg in [
            (float(map_df["lat"].iloc[0]),  float(map_df["lon"].iloc[0]),
             map_df["station_name"].iloc[0],  "top right", C["green"], "Departure"),
            (float(map_df["lat"].iloc[-1]), float(map_df["lon"].iloc[-1]),
             map_df["station_name"].iloc[-1], "top left",  C["red"],   "Arrival"),
        ]:
            fig.add_trace(go.Scattermap(
                lat=[row_lat], lon=[row_lon],
                mode="markers+text",
                marker=dict(size=20, color=col),
                text=[row_name], textposition=tpos,
                textfont=dict(size=11, color=C["dark"],
                              family="Inter, sans-serif"),
                name=leg,
                hovertemplate="<b>%{text}</b><extra></extra>",
            ))

    lat_c = float(map_df["lat"].mean()) if not map_df.empty else 50.0
    lon_c = float(map_df["lon"].mean()) if not map_df.empty else 15.5
    fig.update_layout(
        map=dict(style="open-street-map",
                 center=dict(lat=lat_c, lon=lon_c), zoom=7),
        height=540, margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor="#E2E8F0",
                    borderwidth=1, x=0.01, y=0.99, font=dict(size=11)),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DYPOD Simulator", page_icon="🚆", layout="wide"
)
st.markdown("""<style>
[data-testid="stSidebar"]{background:#F1F5F9}
.stButton>button{border-radius:8px;font-weight:600;transition:all .15s}
.stButton>button:hover{transform:translateY(-1px);box-shadow:0 2px 8px rgba(0,0,0,.15)}
.block-container{padding-top:1.5rem}
.stTabs [data-baseweb="tab"]{font-weight:600;font-size:.92rem}
h3{color:#1E3A5F;font-size:1.2rem}
.sec{font-size:.92rem;font-weight:700;color:#1E3A5F;
     border-left:3px solid #2563EB;padding-left:8px;margin:14px 0 6px}
.info-box{background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;
          padding:10px 14px;font-size:.85rem;color:#1E40AF;margin:8px 0}
.warn-box{background:#FEF9C3;border:1px solid #FDE047;border-radius:8px;
          padding:10px 14px;font-size:.85rem;color:#713F12;margin:8px 0}
.danger-box{background:#FEF2F2;border:1px solid #FECACA;border-radius:8px;
            padding:10px 14px;font-size:.85rem;color:#7F1D1D;margin:8px 0}
.ok-box{background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;
        padding:10px 14px;font-size:.85rem;color:#14532D;margin:8px 0}
</style>""", unsafe_allow_html=True)

_SS_DEFAULTS = dict(
    parser=None, xml_path=None, profile_df=None, via_ops=[],
    rep_result=None, mc_result=None, elec_analysis=None,
    op_start=None, op_end=None,
)
for k, v in _SS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚆 DYPOD Simulator")
    st.caption("Czech railway track profile & energy analysis")
    st.markdown("---")

    # ── 1. File ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">📂 Load railML</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload railML or zip", type=["xml", "railml", "zip"],
        label_visibility="collapsed",
    )
    xml_path = None
    if uploaded:
        xp = load_xml_from_upload(uploaded)
        if xp and xp != st.session_state.xml_path:
            for k, v in _SS_DEFAULTS.items():
                st.session_state[k] = v
            st.session_state.xml_path = xp
        xml_path = st.session_state.xml_path
    else:
        local = sorted(
            glob.glob("*.xml") + glob.glob("*.railml") + glob.glob("*.zip") +
            glob.glob("/tmp/*.xml") + glob.glob("/tmp/*.railml")
        )
        if local:
            sel = st.selectbox("Or pick local file", local,
                               label_visibility="collapsed")
            if sel != st.session_state.xml_path:
                for k, v in _SS_DEFAULTS.items():
                    st.session_state[k] = v
                st.session_state.xml_path = sel
            xml_path = sel

    @st.cache_resource(show_spinner="Parsing railML infrastructure…")
    def _load(path: str) -> DYPODParser:
        if path.lower().endswith(".zip"):
            tmp = tempfile.mkdtemp()
            with zipfile.ZipFile(path) as zf:
                xmls = [n for n in zf.namelist()
                        if n.lower().endswith((".xml", ".railml"))]
                zf.extract(xmls[0], tmp)
                path = os.path.join(tmp, xmls[0])
        return DYPODParser(path)

    parser: DYPODParser | None = None
    if xml_path:
        try:
            parser = _load(xml_path)
            st.success(
                f"✅ Parsed in {parser.parse_time} s\n\n"
                f"**{len(parser.op_info):,}** OPs · "
                f"**{len(parser.seg_props):,}** segments · "
                f"**{len(parser.station_list):,}** passenger stops"
            )
        except Exception as exc:
            st.error(f"Parse error: {exc}")
            st.stop()
    else:
        st.info("Upload a DYPOD railML file to begin.")
        st.stop()

    # ── 2. Route with live search suggestions ─────────────────────────────────
    st.markdown('<div class="sec">🗺️ Route</div>', unsafe_allow_html=True)

    def _station_selector(label: str, key_q: str, key_sel: str) -> str | None:
        """Text input + filtered selectbox."""
        query = st.text_input(label, key=key_q, placeholder="Type to search…")
        results = parser.search_stations(query)
        if not results:
            if query.strip():
                st.caption("⚠️ No matching stations")
            return None
        options = {f"{nm}": oid for oid, nm in results}
        chosen = st.selectbox(
            f"↳ {label}", options=list(options.keys()),
            key=key_sel, label_visibility="collapsed",
        )
        return options.get(chosen)

    op_start = _station_selector("🔍 Departure station", "q_dep", "sel_dep")
    op_end   = _station_selector("🔍 Arrival station",   "q_arr", "sel_arr")

    st.markdown("**⚡ Electrified route preference**")
    elec_reroute = st.toggle(
        "Prefer electrified track", value=False,
        help="Adds a cost penalty to non-electrified segments so the router "
             "finds an all-electric path where possible.",
    )
    pen_km = 100
    if elec_reroute:
        pen_km = st.slider(
            "Detour tolerance (km per NONE segment)", 10, 500, 100, 10,
            help="How many extra km are acceptable to avoid 1 non-electrified segment.",
        )

    btn_profile = st.button(
        "🗺️  Build Track Profile", use_container_width=True, type="primary",
        disabled=not (op_start and op_end and op_start != op_end),
    )

    # ── 3. Vehicle ────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">🚃 Vehicle</div>', unsafe_allow_html=True)
    veh = st.selectbox("Preset", ["Custom"] + list(PREDEFINED_VEHICLES.keys()),
                       label_visibility="collapsed")
    if veh == "Custom":
        traction   = st.selectbox("Traction", ["DIESEL", "ELECTRIC"])
        cv1, cv2   = st.columns(2)
        mass       = cv1.number_input("Mass (kg)",    value=100_000, step=5_000)
        length     = cv2.number_input("Length (m)",   value=40,      step=5)
        power      = cv1.number_input("Power (kW)",   value=500,     step=50)
        aux_power  = cv2.number_input("Aux (kW)",     value=40,      step=5)
        accel      = st.slider("Accel (m/s²)",  0.2, 1.5, 0.6, 0.05)
        decel      = st.slider("Brake (m/s²)",  0.4, 1.5, 0.9, 0.05)
        efficiency = st.slider("Efficiency (%)", 15, 95, 35) / 100.0
        diesel_d   = st.number_input("Diesel (kWh/L)", value=10.0, step=0.1)
    else:
        p = PREDEFINED_VEHICLES[veh]
        traction, mass, length = p["traction"], p["mass"], p["length"]
        power, aux_power       = p["power"], p["aux_power"]
        accel, decel, efficiency = p["accel"], p["decel"], p["efficiency"]
        diesel_d = 10.0
    unit_lbl = "L fuel" if traction == "DIESEL" else "kWh"

    # ── 4. Simulation ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec">⚙️ Simulation</div>', unsafe_allow_html=True)
    dwell     = st.number_input("Station dwell (s)", value=30, step=5)
    stop_mode = st.radio(
        "Stop mode",
        options=["all", "random", "none"],
        format_func={"all": "All stops", "random": "Stochastic", "none": "Express"}.get,
        horizontal=True,
    )
    stop_prob = 1.0
    if stop_mode == "random":
        stop_prob = st.slider("Request-stop probability", 0.0, 1.0, 0.5, 0.05)

    coast_threshold_m = 500.0
    if traction == "ELECTRIC":
        st.markdown("**⚡ Electric coasting**")
        coast_km = st.slider(
            "Max coastable non-electrified gap (km)", 0.1, 10.0, 0.5, 0.1,
            help="Electric vehicles coast (no traction, aux only) through "
                 "non-electrified gaps shorter than this. Reflects real-world "
                 "pantograph-down coasting through neutral sections or "
                 "short construction detours.",
        )
        coast_threshold_m = coast_km * 1000.0

    # ── 5. Monte Carlo ─────────────────────────────────────────────────────────
    st.markdown('<div class="sec">🎲 Monte Carlo</div>', unsafe_allow_html=True)
    mc_n     = st.number_input("Runs per probability", 20, 500, 100, 10)
    mc_probs = st.multiselect(
        "Probabilities to sweep",
        options=[1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
        default=[1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
    )

    cb1, cb2 = st.columns(2)
    btn_run = cb1.button("▶️ Run",  use_container_width=True,
                          disabled=st.session_state.profile_df is None)
    btn_mc  = cb2.button("🎲 MC",   use_container_width=True,
                          disabled=st.session_state.profile_df is None)


# ─── Actions ─────────────────────────────────────────────────────────────────
if btn_profile and op_start and op_end:
    pen = pen_km * 1000.0 if elec_reroute else 0.0
    with st.spinner("Finding path and building profile…"):
        ea  = parser.analyse_electrification(op_start, op_end)
        df  = parser.build_profile(op_start, op_end,
                                   via_ops=st.session_state.via_ops or [],
                                   unelec_penalty_m=pen)
    if df is None or df.empty:
        st.error("No path found between the selected stations. "
                 "They may not be connected in this export.")
    else:
        st.session_state.update(
            profile_df=df, op_start=op_start, op_end=op_end,
            elec_analysis=ea, rep_result=None, mc_result=None,
        )

if btn_run and st.session_state.profile_df is not None:
    with st.spinner("Running physics simulation…"):
        try:
            track = TrackProfile(st.session_state.profile_df)
            sim   = TrainSimulator(mass, length, power, aux_power,
                                   accel, decel, traction, efficiency,
                                   coast_threshold_m=coast_threshold_m)
            hist, snames, stats = sim.run(
                track, stop_mode=stop_mode, stop_prob=stop_prob,
                dwell=dwell, record=True,
            )
            st.session_state.rep_result = dict(
                hist=hist, stop_names=snames, stats=stats,
                traction=traction, diesel_d=diesel_d, efficiency=efficiency,
                unit_lbl=unit_lbl, vehicle_name=veh, dwell=dwell,
                coast_threshold_m=coast_threshold_m,
            )
        except RuntimeError as e:
            st.error(str(e))

if btn_mc and st.session_state.profile_df is not None and mc_probs:
    try:
        track = TrackProfile(st.session_state.profile_df)
        sim   = TrainSimulator(mass, length, power, aux_power,
                               accel, decel, traction, efficiency,
                               coast_threshold_m=coast_threshold_m)
        rows_mc = []

        # Interactive Progress Bar so the UI doesn't silently freeze
        prog_bar = st.progress(0.0, text="Initializing Monte Carlo Engine...")
        total_runs = len(mc_probs) * int(mc_n)
        completed_runs = 0

        for p_val in sorted(mc_probs, reverse=True):
            sm = ("all" if p_val == 1.0 else "none" if p_val == 0.0 else "random")
            e_list, t_list = [], []
            for _ in range(int(mc_n)):
                _, _, st_ = sim.run(track, sm, p_val, dwell)
                e_list.append(to_unit(st_, traction, diesel_d, efficiency))
                t_list.append(st_["journey_time_s"])

                completed_runs += 1
                if completed_runs % max(1, total_runs // 20) == 0:
                    prog_bar.progress(completed_runs / total_runs, text=f"Computing permutations ({completed_runs}/{total_runs})")

            rows_mc.append(dict(
                prob=f"{int(p_val * 100)}%", p_num=p_val,
                mean_e=np.mean(e_list),  std_e=np.std(e_list),
                min_e=np.min(e_list),    max_e=np.max(e_list),
                mean_t=np.mean(t_list),  min_t=np.min(t_list), max_t=np.max(t_list),
            ))
        prog_bar.empty()

        mc_df = pd.DataFrame(rows_mc)
        mc_df["savings"] = (mc_df["mean_e"].max() - mc_df["mean_e"]).clip(lower=0)
        mc_df["unit"]    = unit_lbl
        st.session_state.mc_result = mc_df
    except RuntimeError as e:
        st.error(str(e))


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_prof, tab_edit, tab_run_t, tab_mc_t = st.tabs([
    "🗺️ Infrastructure Limits", "✏️ Profile Editor", "▶️ Kinematic Simulation", "🎲 Monte Carlo",
])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – TRACK PROFILE (INFRASTRUCTURE LIMITS)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_prof:
    df  = st.session_state.profile_df
    ea  = st.session_state.elec_analysis

    if df is None:
        st.markdown("### 👈 Search for stations in the sidebar and click **Build Track Profile**")
        st.info(
            "The tool stitches meso-level railML segments into a direction-aware "
            "track profile including speed limits (per-direction minimum across all "
            "track sub-elements), gradient, electrification, and GPS coordinates."
        )
        st.stop()

    total_km = float(df["cum_km"].max())
    stops_x  = df[df["stop_type"] == "X"]
    stops_r  = df[df["stop_type"] == "R"]
    ue_km    = df[df["electrification"] == "NONE"]["length_m"].sum() / 1000
    elec_km  = total_km - ue_km
    max_spd  = df["speed_kmh"].max()
    max_grad = df["gradient_perm"].abs().max()
    sn       = df["station_name"].iloc[0]
    en       = df["station_name"].iloc[-1]

    st.markdown(f"### 📍 {sn}  →  {en}")
    st.caption("⚠️ **Note:** The chart below shows the maximum allowed track speed (statutory limits). To view the train's actual acceleration and braking curves, run a simulation and check the **▶️ Kinematic Simulation** tab.")

    # Electrification alerts
    if ea is not None and traction == "ELECTRIC":
        ue        = ea["normal_ue_km"]
        gw        = ea["gateway_km"]
        coast_lbl = f"{coast_threshold_m / 1000:.1f} km"
        if ue == 0:
            st.markdown('<div class="ok-box">✅ <b>Fully electrified route</b> — electric vehicle can run without restriction.</div>', unsafe_allow_html=True)
        elif ue <= coast_threshold_m / 1000:
            st.markdown(f'<div class="ok-box">⚡ <b>Short non-electrified gap: {ue:.2f} km</b> — within coasting limit ({coast_lbl}). The vehicle coasts on inertia through this section (aux power only, no traction or regen).</div>', unsafe_allow_html=True)
        elif gw > 0 and abs(gw - ue) < 0.1:
            st.markdown(f'<div class="warn-box">⚠️ <b>Unavoidable non-electrified run-in: {gw:.1f} km</b> (coasting limit: {coast_lbl}). No electrified exit from <b>{sn}</b>. Raise the coasting limit or use a diesel vehicle.</div>', unsafe_allow_html=True)
        elif ea.get("has_alternative") and not elec_reroute:
            st.markdown(f'<div class="warn-box">⚡ <b>{ue:.1f} km non-electrified</b> on this route ({gw:.1f} km unavoidable run-in). Enable <b>Prefer electrified route</b> to save <b>{ea["elec_saving_km"]:.1f} km</b> (+{ea["detour_km"]:.0f} km detour). Or raise coasting limit (now {coast_lbl}).</div>', unsafe_allow_html=True)
        elif elec_reroute and ea.get("has_alternative"):
            st.markdown(f'<div class="ok-box">✅ <b>Electrified routing active.</b> Saved <b>{ea["elec_saving_km"]:.1f} km</b> non-electrified (+{ea["detour_km"]:.0f} km detour). Remaining: <b>{ea["penalised_ue_km"]:.1f} km</b> unavoidable (coast limit: {coast_lbl}).</div>', unsafe_allow_html=True)
        elif ue > 0:
            st.markdown(f'<div class="danger-box">🚫 <b>{ue:.1f} km non-electrified</b> exceeds coasting limit ({coast_lbl}). Raise the limit, use diesel, or choose different stations.</div>', unsafe_allow_html=True)

    # KPI row
    kc = st.columns(6)
    kc[0].markdown(kpi_card(f"{total_km:.1f} km",       "Route length"),            unsafe_allow_html=True)
    kc[1].markdown(kpi_card(str(len(stops_x)),           "Mandatory stops (X)"),    unsafe_allow_html=True)
    kc[2].markdown(kpi_card(str(len(stops_r)),           "Request halts (R)"),      unsafe_allow_html=True)
    kc[3].markdown(kpi_card(f"{max_spd:.0f} km/h",      "Max speed"),              unsafe_allow_html=True)
    kc[4].markdown(kpi_card(f"{max_grad:.0f} ‰",        "Max gradient"),           unsafe_allow_html=True)
    kc[5].markdown(kpi_card(
        f"{elec_km:.0f} km", f"Electrified ({100 * elec_km / total_km:.0f}%)",
        delta=(f"⚠️ {ue_km:.0f} km non-electrified" if ue_km > 0 else "✅ Fully electrified"),
        color=(C["red"] if ue_km > 0 else C["green"]),
    ), unsafe_allow_html=True)
    st.write("")

    # Main chart
    st.plotly_chart(make_profile_chart(df), use_container_width=True)

    # Electrification legend badges
    badges = ""
    for sys in df["electrification"].unique():
        col   = elec_color(sys)
        km_e  = df[df["electrification"] == sys]["length_m"].sum() / 1000
        icon  = "⚡" if sys != "NONE" else "🛢️"
        badges += (f'<span style="background:{col};color:white;padding:3px 10px;'
                   f'border-radius:12px;font-size:.8rem;font-weight:600;margin:2px;'
                   f'display:inline-block">{icon} {sys} · {km_e:.1f} km</span> ')
    st.markdown(badges, unsafe_allow_html=True)
    st.write("")

    # Map
    map_df = df.dropna(subset=["lat", "lon"])
    if not map_df.empty:
        with st.expander("🗺️ Route map", expanded=True):
            st.plotly_chart(
                make_route_map(df, st.session_state.via_ops or [], parser),
                use_container_width=True,
            )

    # Gradient stats + histogram
    with st.expander("📐 Gradient statistics"):
        g_vals = df["gradient_perm"]
        gc1, gc2 = st.columns([2, 1])
        with gc1:
            fig_gh = go.Figure()
            fig_gh.add_trace(go.Histogram(
                x=g_vals[g_vals >= 0], name="Uphill",
                marker_color=C["red"], opacity=0.75, nbinsx=25))
            fig_gh.add_trace(go.Histogram(
                x=g_vals[g_vals < 0], name="Downhill",
                marker_color=C["primary"], opacity=0.75, nbinsx=25))
            fig_gh.update_layout(
                barmode="overlay", height=260,
                xaxis_title="Gradient [‰]", yaxis_title="Segments",
                paper_bgcolor="white", plot_bgcolor="white",
                legend=dict(orientation="h"),
                margin=dict(t=10, b=40, l=50, r=10),
            )
            st.plotly_chart(fig_gh, use_container_width=True)
        with gc2:
            st.metric("Max uphill",   f"{g_vals.max():.1f} ‰")
            st.metric("Max downhill", f"{g_vals.min():.1f} ‰")
            st.metric("RMS gradient", f"{(g_vals**2).mean()**0.5:.1f} ‰")
            st.metric("Mean",         f"{g_vals.mean():.1f} ‰")

    # Speed profile summary — pure Plotly, no matplotlib dependency
    with st.expander("⚡ Speed profile summary"):
        spd_df = df[["cum_km", "station_name", "speed_kmh", "gradient_perm",
                      "electrification", "n_tracks", "stop_type"]].copy()
        spd_df = spd_df[spd_df["station_name"].str.strip().ne("")].reset_index(drop=True)

        def _spd_cell_color(v):
            lo, hi = 30.0, 160.0
            t = max(0.0, min(1.0, (float(v) - lo) / (hi - lo)))
            r = int(220 * (1 - t) + 34 * t)
            g = int(38  * (1 - t) + 197 * t)
            b = int(38  * (1 - t) + 94 * t)
            return f"rgba({r},{g},{b},0.25)"

        stop_icons = spd_df["stop_type"].map({"X": "🚉", "R": "🛑"}).fillna("")
        labels = stop_icons + " " + spd_df["station_name"].str.strip()

        fig_tbl = go.Figure(go.Table(
            columnwidth=[60, 200, 80, 70, 160, 70],
            header=dict(
                values=["km", "Waypoint", "Speed [km/h]", "Grad [‰]",
                        "Electrification", "Tracks"],
                fill_color=C["dark"], font=dict(color="white", size=12),
                align="left", height=28,
            ),
            cells=dict(
                values=[
                    spd_df["cum_km"].round(2),
                    labels,
                    spd_df["speed_kmh"].astype(int),
                    spd_df["gradient_perm"].round(1),
                    spd_df["electrification"],
                    spd_df["n_tracks"],
                ],
                fill_color=[
                    ["#F8FAFC"] * len(spd_df),
                    ["#F8FAFC"] * len(spd_df),
                    [_spd_cell_color(v) for v in spd_df["speed_kmh"]],
                    ["#F8FAFC"] * len(spd_df),
                    [elec_color(e) + "44" for e in spd_df["electrification"]],
                    ["#F8FAFC"] * len(spd_df),
                ],
                font=dict(size=11), align="left", height=24,
            ),
        ))
        fig_tbl.update_layout(
            height=min(60 + len(spd_df) * 26, 460),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig_tbl, use_container_width=True)

    # Full data table
    with st.expander("📋 Full profile data table"):
        cols = ["cum_km", "station_name", "stop_type", "speed_kmh",
                "gradient_perm", "electrification", "recuperation",
                "length_m", "ue_gap_m", "n_tracks", "op_types"]
        cols = [c for c in cols if c in df.columns]
        ren  = dict(cum_km="km [km]", station_name="Waypoint", stop_type="Stop",
                    speed_kmh="Speed [km/h]", gradient_perm="Grad [‰]",
                    electrification="Electrif.", recuperation="Recup.",
                    length_m="Length [m]", ue_gap_m="UE gap [m]",
                    n_tracks="Tracks", op_types="Type")
        st.dataframe(df[cols].rename(columns=ren),
                     use_container_width=True, hide_index=True, height=350)
        st.download_button("⬇️ Download profile CSV",
                            df[cols].to_csv(index=False).encode(),
                            "track_profile.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – PROFILE EDITOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_edit:
    st.markdown("### ✏️ Interactive Profile Editor")
    df = st.session_state.profile_df

    # Via waypoints
    st.markdown('<div class="sec">📍 Via Waypoints</div>', unsafe_allow_html=True)
    st.caption("Force the route through specific intermediate stations. "
               "Rebuild the profile after adding or removing waypoints.")

    qv     = st.text_input("Search waypoint to add", placeholder="e.g. Pardubice hl.n.", key="q_via")
    res_v  = parser.search_stations(qv) if qv.strip() else []
    cva, cvb = st.columns([3, 1])
    via_sel  = None
    if res_v:
        via_opts = {nm: oid for oid, nm in res_v}
        via_sel  = cva.selectbox("Select", list(via_opts.keys()),
                                  label_visibility="collapsed", key="via_opt")
    add_via = cvb.button("➕ Add", use_container_width=True, disabled=not res_v)
    if add_via and via_sel and res_v:
        via_op = via_opts.get(via_sel)
        if via_op and via_op not in st.session_state.via_ops:
            st.session_state.via_ops.append(via_op)
            st.rerun()

    if st.session_state.via_ops:
        st.markdown("**Via waypoints (in order):**")
        for i, vid in enumerate(st.session_state.via_ops):
            info = parser.op_info.get(vid, {})
            e1, e2, e3, e4 = st.columns([0.25, 0.25, 3, 1])
            e1.markdown(f"**{i + 1}**")
            if e2.button("↑", key=f"up_{i}", disabled=i == 0):
                st.session_state.via_ops[i], st.session_state.via_ops[i - 1] = (
                    st.session_state.via_ops[i - 1], st.session_state.via_ops[i])
                st.rerun()
            e3.markdown(
                f"📍 **{info.get('name', vid)}** "
                f"<span style='color:{C['grey']};font-size:.8rem'>"
                f"{', '.join(info.get('types', []))}</span>",
                unsafe_allow_html=True,
            )
            if e4.button("✕ Remove", key=f"rm_{i}", use_container_width=True):
                st.session_state.via_ops.pop(i)
                st.rerun()
        if st.button("🗑️ Clear all"):
            st.session_state.via_ops = []
            st.rerun()
        st.markdown('<div class="info-box">ℹ️ Click <b>Build Track Profile</b> in the sidebar to apply waypoints.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">No via-waypoints set. The router uses the direct shortest path.</div>', unsafe_allow_html=True)

    # Editable overrides
    if df is not None:
        st.markdown('<div class="sec">📝 Speed & Stop Overrides</div>', unsafe_allow_html=True)
        st.caption(
            "Edit speed limits or stop types directly. "
            "Changes are applied to the simulation profile only — "
            "the underlying railML data is never modified."
        )
        edit_cols = ["station_name", "stop_type", "speed_kmh",
                     "gradient_perm", "electrification", "length_m"]
        edit_cols = [c for c in edit_cols if c in df.columns]
        edited = st.data_editor(
            df[edit_cols].copy(),
            column_config={
                "station_name":    st.column_config.TextColumn("Waypoint",      disabled=True, width="large"),
                "stop_type":       st.column_config.SelectboxColumn("Stop",    options=["X", "R", ""], width="small"),
                "speed_kmh":       st.column_config.NumberColumn("Speed [km/h]", min_value=0, max_value=350, width="small"),
                "gradient_perm":   st.column_config.NumberColumn("Grad [‰]",  disabled=True, width="small"),
                "electrification": st.column_config.TextColumn("Electrif.",   disabled=True, width="medium"),
                "length_m":        st.column_config.NumberColumn("Length [m]", disabled=True, width="small"),
            },
            use_container_width=True, hide_index=True,
            num_rows="fixed", key="editor_tbl",
        )
        if st.button("✅ Apply overrides", type="primary"):
            updated = df.copy()
            updated["speed_kmh"] = edited["speed_kmh"].values
            updated["stop_type"] = edited["stop_type"].values
            st.session_state.profile_df = updated
            st.session_state.rep_result = None
            st.success("✅ Profile updated. Run the simulation from the **▶️ Kinematic Simulation** tab.")
    else:
        st.info("Build a track profile first (sidebar → **Build Track Profile**).")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 – KINEMATIC SIMULATION (REPRESENTATIVE RUN)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_run_t:
    rep = st.session_state.rep_result
    if rep is None:
        st.info("👈 Build a profile, then click **▶️ Run** in the sidebar.")
        st.stop()

    hist   = rep["hist"]
    stats  = rep["stats"]
    snames = rep["stop_names"]
    _tr    = rep["traction"]
    _dd    = rep["diesel_d"]
    _eff   = rep["efficiency"]
    consumed = to_unit(stats, _tr, _dd, _eff)
    df_p   = st.session_state.profile_df
    tot_km = float(df_p["cum_km"].max()) if df_p is not None else 0
    avg_spd = tot_km / (stats["journey_time_s"] / 3600) if stats["journey_time_s"] > 0 else 0
    sn_r   = df_p["station_name"].iloc[0]  if df_p is not None else ""
    en_r   = df_p["station_name"].iloc[-1] if df_p is not None else ""

    st.markdown(f"### ▶️ {sn_r}  →  {en_r}  ·  {rep['vehicle_name']}")

    kc2 = st.columns(6)
    kc2[0].markdown(kpi_card(fmt_dur(stats["journey_time_s"]), "Journey time"),             unsafe_allow_html=True)
    kc2[1].markdown(kpi_card(f"{consumed:.1f}",                f"Net consumed [{unit_lbl}]"), unsafe_allow_html=True)
    kc2[2].markdown(kpi_card(f"{stats['gross_kwh']:.1f}",      "Gross energy [kWh]"),        unsafe_allow_html=True)
    kc2[3].markdown(kpi_card(f"{stats['regen_kwh']:.1f}",      "Recuperated [kWh]"),         unsafe_allow_html=True)
    kc2[4].markdown(kpi_card(f"{avg_spd:.1f} km/h",            "Avg journey speed"),         unsafe_allow_html=True)
    kc2[5].markdown(kpi_card(str(len(snames)),                 "Stops served"),              unsafe_allow_html=True)
    st.write("")

    if snames:
        st.markdown("**Stops served:** " + "  →  ".join(f"*{s}*" for s in snames))
    else:
        st.caption("No intermediate stops served on this run.")

    if hist:
        fig3 = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("Simulated Train Speed [km/h]", "Cumulative Energy [kWh]"),
            vertical_spacing=0.10, row_heights=[0.55, 0.45],
        )
        # Note for long routes:
        if tot_km > 30:
            st.caption("🔍 **Tip:** On routes longer than 30km, the acceleration/braking curves might appear vertically squashed. Drag a box over the chart to zoom in and see the detailed train kinematics.")

        fig3.add_trace(go.Scatter(
            x=hist["km"], y=hist["v_limit_kmh"], name="Track Limit",
            line=dict(color=C["red"], dash="dot", width=1.8, shape="hv")),
            row=1, col=1)
        fig3.add_trace(go.Scatter(
            x=hist["km"], y=hist["v_kmh"], name="Train Speed",
            line=dict(color=C["primary"], width=2.5),
            fill="tozeroy", fillcolor=C["bg_blue"]), row=1, col=1)
        fig3.add_trace(go.Scatter(
            x=hist["km"], y=hist["gross_kwh"], name="Gross energy",
            line=dict(color=C["yellow"], width=2)), row=2, col=1)
        fig3.add_trace(go.Scatter(
            x=hist["km"], y=hist["regen_kwh"], name="Recuperated",
            line=dict(color=C["green"], width=2, dash="dash")), row=2, col=1)
        fig3.add_trace(go.Scatter(
            x=hist["km"], y=hist["net_kwh"], name="Net energy",
            line=dict(color=C["secondary"], width=2.5)), row=2, col=1)

        if df_p is not None:
            for _, sr in df_p[df_p["stop_type"] == "X"].iterrows():
                fig3.add_vline(x=sr["cum_km"], line_width=0.7,
                               line_dash="dot", line_color="#CBD5E1", row=1, col=1)

        fig3.update_xaxes(title_text="Distance from departure [km]", row=2, col=1,
                          gridcolor=C["light"])
        fig3.update_yaxes(title_text="Speed [km/h]",  row=1, col=1, gridcolor=C["light"])
        fig3.update_yaxes(title_text="Energy [kWh]",  row=2, col=1, gridcolor=C["light"])
        fig3.update_layout(
            height=580, paper_bgcolor="white", plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1),
            margin=dict(t=70, b=10, l=60, r=20),
            font=dict(family="Inter, sans-serif", size=12),
        )
        st.plotly_chart(fig3, use_container_width=True)

        if stats["gross_kwh"] > 0 and tot_km > 0:
            rp   = stats["regen_kwh"] / stats["gross_kwh"] * 100
            spec = stats["net_kwh"] / tot_km
            st.markdown("---")
            ec1, ec2, ec3, ec4 = st.columns(4)
            ec1.metric("Recuperation share",  f"{rp:.1f}%")
            ec2.metric("Specific net energy", f"{spec:.3f} kWh/km")
            if _tr == "DIESEL":
                ec3.metric("Specific fuel", f"{consumed / tot_km * 1000:.1f} mL/km")
                ec4.metric("CO₂ (est.)",    f"{consumed * 2.65:.0f} kg",
                            help="Diesel combustion: ~2.65 kg CO₂/L")
            else:
                ec3.metric("Net kWh/km", f"{spec:.3f}")
                ec4.metric("Regen saved", f"{stats['regen_kwh']:.1f} kWh")

        with st.expander("📊 Detailed distributions"):
            dd1, dd2 = st.columns(2)
            with dd1:
                fig_sv = go.Figure(go.Histogram(
                    x=hist["v_kmh"], nbinsx=40,
                    marker_color=C["primary"], opacity=0.8))
                fig_sv.update_layout(
                    title="Time distribution at speed",
                    xaxis_title="Speed [km/h]", yaxis_title="Seconds",
                    height=280, paper_bgcolor="white", plot_bgcolor="white",
                    margin=dict(t=40, b=40, l=50, r=10))
                st.plotly_chart(fig_sv, use_container_width=True)
            with dd2:
                wf_v = [stats["gross_kwh"], -stats["regen_kwh"], stats["net_kwh"]]
                fig_wf = go.Figure(go.Waterfall(
                    orientation="v",
                    measure=["absolute", "relative", "total"],
                    x=["Gross consumed", "Recuperated", "Net consumed"],
                    y=wf_v,
                    connector=dict(line=dict(color="#CBD5E1")),
                    decreasing=dict(marker=dict(color=C["green"])),
                    increasing=dict(marker=dict(color=C["red"])),
                    totals=dict(marker=dict(color=C["primary"])),
                    text=[f"{v:.1f} kWh" for v in wf_v],
                    textposition="outside",
                ))
                fig_wf.update_layout(
                    title="Energy waterfall [kWh]", height=280,
                    paper_bgcolor="white", plot_bgcolor="white",
                    margin=dict(t=40, b=40, l=50, r=10))
                st.plotly_chart(fig_wf, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 – MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mc_t:
    mc_df = st.session_state.mc_result
    df_p  = st.session_state.profile_df

    if mc_df is None:
        st.info("👈 Build a profile, then click **🎲 MC** in the sidebar.")
        if df_p is not None:
            tr = TrackProfile(df_p)
            if tr.n_request == 0:
                st.markdown(
                    '<div class="warn-box">⚠️ <b>No request stops (R)</b> on this route — '
                    'Monte Carlo variance will be zero. <br>'
                    'All stops are mandatory stations (X). For meaningful stochastic '
                    'results choose a route containing <i>stoppingPoint</i>-type halts '
                    '(shown as R), which are served with configurable probability.</div>',
                    unsafe_allow_html=True,
                )
        st.stop()

    ul     = mc_df["unit"].iloc[0]
    route_n = (f"{df_p['station_name'].iloc[0]} → {df_p['station_name'].iloc[-1]}"
               if df_p is not None else "")

    st.markdown(f"### 🎲 Monte Carlo — {route_n}")

    if df_p is not None:
        tr = TrackProfile(df_p)
        nm, nr = tr.n_mandatory, tr.n_request
        if nr == 0:
            st.markdown(
                '<div class="warn-box">⚠️ All stops are mandatory (X) — std = 0. '
                'Try a route with stoppingPoint-type halts.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="info-box">ℹ️ Route has <b>{nm} mandatory (X)</b> and '
                f'<b>{nr} request (R)</b> stops. MC sweeps the probability of '
                f'serving request halts.</div>',
                unsafe_allow_html=True)

    st.caption(f"N = {int(mc_n)} simulation runs per probability level")

    # Summary table
    disp = mc_df.copy()
    disp["Mean time"] = disp["mean_t"].apply(fmt_dur)
    disp["Fastest"]   = disp["min_t"].apply(fmt_dur)
    disp["Slowest"]   = disp["max_t"].apply(fmt_dur)
    show_map = {
        "prob": "Probability",
        "mean_e": f"Mean [{ul}]",  "std_e": f"Std [{ul}]",
        "min_e":  f"Min [{ul}]",   "max_e": f"Max [{ul}]",
        "savings": f"Savings [{ul}]",
        "Mean time": "Mean time",  "Fastest": "Fastest", "Slowest": "Slowest",
    }
    disp2 = disp.rename(columns=show_map)
    st.dataframe(
        disp2[[v for v in show_map.values() if v in disp2.columns]],
        use_container_width=True, hide_index=True,
    )
    st.download_button("⬇️ Download CSV",
                        mc_df.to_csv(index=False).encode(),
                        "mc_results.csv", "text/csv")
    st.markdown("---")

    # Charts
    mc1, mc2 = st.columns(2)

    with mc1:
        fig_bar = go.Figure(go.Bar(
            x=mc_df["prob"], y=mc_df["savings"],
            marker=dict(color=mc_df["savings"], colorscale="Blues",
                        showscale=True, colorbar=dict(title=ul, len=0.8)),
            text=mc_df["savings"].round(2),
            texttemplate="%{text}", textposition="outside",
            hovertemplate="<b>%{x}</b><br>Savings: %{y:.2f} " + ul + "<extra></extra>",
        ))
        fig_bar.update_layout(
            title=f"<b>Energy savings vs. all-stops</b> [{ul}]",
            xaxis_title="Request-stop probability",
            yaxis_title=f"Savings [{ul}]",
            height=400, paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(t=60, b=50, l=60, r=20),
            font=dict(family="Inter, sans-serif", size=12),
            yaxis=dict(gridcolor=C["light"]),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with mc2:
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(
            x=list(mc_df["prob"]) + list(mc_df["prob"].iloc[::-1]),
            y=list(mc_df["max_e"]) + list(mc_df["min_e"].iloc[::-1]),
            fill="toself", fillcolor=C["bg_blue"],
            line=dict(color="rgba(255,255,255,0)"), name="Min–Max range",
        ))
        fig_err.add_trace(go.Scatter(
            x=list(mc_df["prob"]) + list(mc_df["prob"].iloc[::-1]),
            y=list(mc_df["mean_e"] + mc_df["std_e"]) +
              list((mc_df["mean_e"] - mc_df["std_e"]).iloc[::-1]),
            fill="toself", fillcolor="rgba(37,99,235,0.18)",
            line=dict(color="rgba(255,255,255,0)"), name="Mean ± 1σ",
        ))
        fig_err.add_trace(go.Scatter(
            x=mc_df["prob"], y=mc_df["mean_e"],
            mode="markers+lines",
            marker=dict(size=11, color=C["primary"], line=dict(color="white", width=2)),
            line=dict(color=C["primary"], width=2.5), name="Mean",
        ))
        fig_err.update_layout(
            title=f"<b>Energy distribution</b> [{ul}]",
            xaxis_title="Request-stop probability",
            yaxis_title=f"Energy [{ul}]",
            height=400, paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(t=60, b=50, l=60, r=20),
            font=dict(family="Inter, sans-serif", size=12),
            yaxis=dict(gridcolor=C["light"]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_err, use_container_width=True)

    # Journey time
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(
        x=list(mc_df["prob"]) + list(mc_df["prob"].iloc[::-1]),
        y=list(mc_df["max_t"] / 60) + list(mc_df["min_t"].iloc[::-1] / 60),
        fill="toself", fillcolor=C["bg_orange"],
        line=dict(color="rgba(255,255,255,0)"), name="Min–Max range",
    ))
    fig_t.add_trace(go.Scatter(
        x=mc_df["prob"], y=mc_df["mean_t"] / 60,
        mode="markers+lines",
        marker=dict(size=11, color=C["accent"], line=dict(color="white", width=2)),
        line=dict(color=C["accent"], width=2.5), name="Mean time",
    ))
    fig_t.update_layout(
        title="<b>Journey time vs. stopping policy</b>",
        xaxis_title="Request-stop probability",
        yaxis_title="Journey time [min]",
        height=360, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(t=60, b=50, l=60, r=20),
        font=dict(family="Inter, sans-serif", size=12),
        yaxis=dict(gridcolor=C["light"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_t, use_container_width=True)

    with st.expander("📊 Energy–time trade-off"):
        fig_et = go.Figure(go.Scatter(
            x=mc_df["mean_t"] / 60, y=mc_df["mean_e"],
            mode="markers+text",
            marker=dict(
                size=mc_df["savings"].clip(lower=0) * 3 + 14,
                color=mc_df["savings"], colorscale="RdYlGn", showscale=True,
                colorbar=dict(title=f"Savings [{ul}]"),
                line=dict(color="white", width=1.5),
            ),
            text=mc_df["prob"], textposition="top center",
            hovertemplate=(
                "<b>%{text}</b><br>Time: %{x:.1f} min<br>"
                "Energy: %{y:.2f} " + ul + "<extra></extra>"
            ),
        ))
        fig_et.update_layout(
            title="<b>Energy–time trade-off</b> by stopping policy",
            xaxis_title="Mean journey time [min]",
            yaxis_title=f"Mean energy [{ul}]",
            height=420, paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(t=60, b=50, l=60, r=20),
            font=dict(family="Inter, sans-serif", size=12),
        )
        st.plotly_chart(fig_et, use_container_width=True)
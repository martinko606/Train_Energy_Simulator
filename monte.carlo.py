"""
DYPOD Track Profile Builder & Fleet Energy Simulator — Combined Edition
=======================================================================
• Parses DYPOD railML 3.2 (Správa železnic / OLTIS Group)
• Builds direction-aware stitched track profiles
  - Speed: min across all sub-track speedSections per direction
  - Gradient: normal/reverse from gradientCurve applicationDirection
  - Electrification & recuperation per segment
• Live station search with dropdown suggestions
• Interactive profile editor with via-waypoints + table overrides
• Smart electrified-route search (penalty-based rerouting for incompatible catenary)
• Multi-Leg Scenario Builder (Round-Trips, Custom Itineraries)
• Electric coasting through short incompatible/non-electrified gaps (configurable)
• Physics simulation with dynamic Davis equation & vehicle max-speed cap
• Kinematic chart with time/distance X-axis toggle, gradient ribbon & station annotations
• Monte Carlo stochastic stop-probability analysis with progress bar
Run:  streamlit run app.py
"""
from __future__ import annotations
import glob, heapq, itertools, math, os, random, tempfile, time, zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np, pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

# ── vehicle fleet (merged from all codebase versions) ─────────────────────────
PREDEFINED_VEHICLES: dict[str, dict] = {
    "EDITA (diesel railcar)":      dict(traction="DIESEL",  mass=22_000,  length=15,     power=250,   aux_power=20, accel=0.5, decel=0.8, efficiency=0.30, max_speed=80, systems=[]),
    "EDITA+Btax":                  dict(traction="DIESEL",  mass=42_000,  length=30,     power=250,   aux_power=25, accel=0.4, decel=0.8, efficiency=0.30, max_speed=80, systems=[]),
    "Regionova (Class 814)":       dict(traction="DIESEL",  mass=39_600,  length=28.44,  power=242,   aux_power=10, accel=0.5, decel=0.8, efficiency=0.30, max_speed=80, systems=[]),
    "RegioNova Duo (Class 840)":   dict(traction="DIESEL",  mass=50_000,  length=25.5,   power=514,   aux_power=25, accel=0.8, decel=0.9, efficiency=0.38, max_speed=100, systems=[]),
    "750-7 + 3 coaches":           dict(traction="DIESEL",  mass=207_000, length=90.16,  power=1_550, aux_power=40, accel=0.4, decel=0.8, efficiency=0.30, max_speed=100, systems=[]),
    "Regiopanter 3-car (Cl. 640)": dict(traction="ELECTRIC",mass=159_000, length=79.4,   power=2_040, aux_power=80, accel=0.8, decel=0.9, efficiency=0.85, max_speed=160, systems=["3,000V/0Hz", "25,000V/50Hz"]),
    "CityElefant (Class 471)":     dict(traction="ELECTRIC",mass=155_000, length=79,     power=2_000, aux_power=80, accel=0.8, decel=0.9, efficiency=0.85, max_speed=160, systems=["3,000V/0Hz"]),
    "Pendolino (Class 680)":       dict(traction="ELECTRIC",mass=380_000, length=157.9,  power=5_500, aux_power=200,accel=0.8, decel=1.0, efficiency=0.87, max_speed=200, systems=["3,000V/0Hz", "25,000V/50Hz", "15,000V/16.7Hz"]),
}


# ─────────────────────────────────────────────────────────────────────────────
#  DYPOD railML PARSER
# ─────────────────────────────────────────────────────────────────────────────
class DYPODParser:
    """
    Parses a DYPOD railML 3.2 export and exposes a graph of the Czech
    railway meso-level network.
    """

    def __init__(self, xml_path: str):
        t0 = time.time()
        self._root = self._strip_ns(ET.parse(xml_path).getroot())
        self._esys:      dict[str, str]             = {}
        self._sub2seg:   dict[str, str]             = {}
        self._seg_speeds:dict[str, dict[str, list]] = defaultdict(lambda: {"normal": [], "reverse": []})
        self._line_max:  dict[str, int]             = {}
        self.op_info:    dict[str, dict]            = {}
        self.seg_props:  dict[str, dict]            = {}
        self.graph:      dict[str, list]            = {}
        self.op_by_name: dict[str, list]            = {}
        self.station_list: list[tuple[str, str]]    = []

        self._parse_esys(); self._parse_sub2seg(); self._parse_speed_sections()
        self._parse_ops();  self._parse_segs();    self._build_graph()
        self._build_indexes()
        self.parse_time = round(time.time() - t0, 2)

    @staticmethod
    def _strip_ns(root: ET.Element) -> ET.Element:
        for el in root.iter():
            if "}" in el.tag: el.tag = el.tag.split("}", 1)[1]
        return root

    def _parse_esys(self):
        for es in self._root.iter("electrificationSystem"):
            v, f = es.get("voltage", "0"), es.get("frequency", "0")
            self._esys[es.get("id", "")] = "NONE" if v == "0" else f"{int(float(v)):,}V/{f}Hz"

    def _parse_sub2seg(self):
        for ne in self._root.iter("netElement"):
            ecu = ne.find("elementCollectionUnordered")
            if ecu is not None:
                for ep in ecu.findall("elementPart"):
                    ref = ep.get("ref", "")
                    if ref: self._sub2seg[ref] = ne.get("id", "")

    def _parse_speed_sections(self):
        for ss in self._root.iter("speedSection"):
            ss_id = ss.get("id", "")
            maxsp = float(ss.get("maxSpeed", 0))
            if maxsp <= 0: continue
            ane = ss.find(".//associatedNetElement")
            if ane is None: continue
            sub_ne = ane.get("netElementRef", "")
            seg_ne = self._sub2seg.get(sub_ne, sub_ne)
            d = "reverse" if "_reverse" in ss_id else "normal"
            self._seg_speeds[seg_ne][d].append(maxsp)

    def _speed_for(self, ne_ref: str, direction: str) -> float:
        vals = self._seg_speeds.get(ne_ref, {}).get(direction, [])
        if vals: return float(min(vals))
        other = "reverse" if direction == "normal" else "normal"
        vals_o = self._seg_speeds.get(ne_ref, {}).get(other, [])
        if vals_o: return float(min(vals_o))
        lp = self._line_max.get(ne_ref, 0)
        return float(lp) if lp > 0 else 30.0

    def _parse_ops(self):
        for op in self._root.iter("operationalPoint"):
            oid = op.get("id", "")
            nm  = op.find("name")
            name = nm.get("name", oid) if nm is not None else oid
            lat = lon = None
            for sl in op.findall("spotLocation"):
                geo = sl.find("geometricCoordinate")
                if geo is not None and geo.get("positioningSystemRef", "") == "gps01":
                    lat, lon = float(geo.get("x", 0)), float(geo.get("y", 0)); break
            types = [x.get("operationalType", "") for x in op.iter("opOperation")]
            self.op_info[oid] = dict(name=name, lat=lat, lon=lon, types=types)

    def _parse_segs(self):
        grad_map: dict[str, dict] = {}
        for gc in self._root.iter("gradientCurve"):
            grad = float(gc.get("gradient", 0))
            loc  = gc.find("linearLocation")
            if loc is None: continue
            app_dir = loc.get("applicationDirection", "normal")
            ane = loc.find("associatedNetElement")
            if ane is None: continue
            ne_ref = ane.get("netElementRef", "")
            if ne_ref not in grad_map:
                grad_map[ne_ref] = {"normal": 0.0, "reverse": 0.0}

            if app_dir in ("normal", "both", ""):
                grad_map[ne_ref]["normal"] = grad
                grad_map[ne_ref]["reverse"] = -grad
            elif app_dir == "reverse":
                grad_map[ne_ref]["reverse"] = grad
                grad_map[ne_ref]["normal"] = -grad

        elec_map: dict[str, str] = {}
        for sec in self._root.iter("electrificationSection"):
            ane = sec.find(".//associatedNetElement")
            esr = sec.find("electrificationSystemRef")
            if ane is None or esr is None: continue
            seg   = self._sub2seg.get(ane.get("netElementRef", ""), ane.get("netElementRef", ""))
            label = self._esys.get(esr.get("ref", ""), "UNKNOWN")
            elec_map[seg] = label if label != "NONE" else elec_map.get(seg, "NONE")

        len_map: dict[str, float] = {}
        for tr in self._root.iter("track"):
            ane = tr.find(".//associatedNetElement"); lel = tr.find("length")
            if ane is None or lel is None: continue
            seg   = self._sub2seg.get(ane.get("netElementRef", ""), ane.get("netElementRef", ""))
            ltype = lel.get("type", ""); val = float(lel.get("value", 0))
            if ltype == "physical" or seg not in len_map: len_map[seg] = val

        for line in self._root.iter("line"):
            ane = line.find(".//associatedNetElement")
            if ane is None: continue
            ne_ref   = ane.get("netElementRef", "")
            lel      = line.find("length")
            length_m = float(lel.get("value", 0)) if lel is not None else len_map.get(ne_ref, 0.0)

            if length_m <= 0.0: length_m = 10.0  # Failsafe

            lp       = line.find("linePerformance")
            ll       = line.find("lineLayout")
            lo       = line.find("lineOperation")
            lp_spd   = int(lp.get("maxSpeed", 0)) if lp is not None else 0
            self._line_max[ne_ref] = lp_spd
            gr       = grad_map.get(ne_ref, {"normal": 0.0, "reverse": 0.0})
            electr = elec_map.get(ne_ref, "NONE")
            self.seg_props[ne_ref] = dict(
                length_m       = length_m,
                speed_normal   = self._speed_for(ne_ref, "normal"),
                speed_reverse  = self._speed_for(ne_ref, "reverse"),
                grad_normal    = gr["normal"],
                grad_reverse   = gr["reverse"],
                electrification= electr,
                recuperation   = 0 if electr == "NONE" else 1,
                n_tracks       = ll.get("numberOfTracks", "single") if ll is not None else "single",
                mode_of_op     = lo.get("modeOfOperation", "") if lo is not None else "",
                braking_dist   = int(lp.get("signalledBrakingDistance", 0)) if lp is not None else 0,
            )

    def _build_graph(self):
        for line in self._root.iter("line"):
            b_el = line.find("beginsInOP"); e_el = line.find("endsInOP")
            ane  = line.find(".//associatedNetElement")
            if b_el is None or e_el is None or ane is None: continue
            b_op, e_op = b_el.get("ref", ""), e_el.get("ref", "")
            ne_ref     = ane.get("netElementRef", "")
            props      = self.seg_props.get(ne_ref, {})
            length_m   = props.get("length_m", 0.0)
            electr     = props.get("electrification", "NONE")
            self.graph.setdefault(b_op, []).append(
                dict(to=e_op, ne_id=ne_ref, length_m=length_m, forward=True,  electr=electr))
            self.graph.setdefault(e_op, []).append(
                dict(to=b_op, ne_id=ne_ref, length_m=length_m, forward=False, electr=electr))

    def _build_indexes(self):
        for oid, info in self.op_info.items():
            self.op_by_name.setdefault(info["name"], []).append(oid)
            if any(t in PASSENGER_TYPES for t in info.get("types", [])):
                self.station_list.append((info["name"], oid))
        self.station_list.sort(key=lambda x: x[0])

    def dijkstra(self, start: str, end: str,
                 penalty_m: float = 0.0, comp_sys: list[str] = None) -> tuple[float | None, list]:
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
                is_incomp = (comp_sys is not None) and (edge["electr"] not in comp_sys)
                pen = penalty_m if is_incomp else 0.0
                heapq.heappush(pq, (
                    cost + edge["length_m"] + pen,
                    next(tb), nxt,
                    path + [dict(from_op=node, **edge)],
                ))
        return None, []

    def analyse_electrification(self, start_op: str, end_op: str, via_ops: list[str] = None, comp_sys: list[str] = None) -> dict:
        waypoints = [start_op] + (via_ops or []) + [end_op]

        p_normal = []
        for i in range(len(waypoints) - 1):
            _, path = self.dijkstra(waypoints[i], waypoints[i + 1], 0, comp_sys)
            if not path:
                return dict(normal_km=0, normal_ue_km=0, gateway_km=0,
                            gateway_op=None, penalised_km=0, penalised_ue_km=0,
                            detour_km=0, elec_saving_km=0, has_alternative=False)
            p_normal.extend(path)

        def _ue_km(path):
            return sum(self.seg_props.get(s["ne_id"], {}).get("length_m", 0)
                       for s in path if (comp_sys is not None and s["electr"] not in comp_sys)) / 1000

        def _tot_km(path):
            return sum(self.seg_props.get(s["ne_id"], {}).get("length_m", 0)
                       for s in path) / 1000

        normal_ue  = _ue_km(p_normal)
        normal_tot = _tot_km(p_normal)

        gateway_km = 0.0
        gateway_op = None
        for s in p_normal:
            if comp_sys is not None and s["electr"] in comp_sys:
                gateway_op = s["from_op"]
                break
            gateway_km += self.seg_props.get(s["ne_id"], {}).get("length_m", 0) / 1000

        p_pen = []
        for i in range(len(waypoints) - 1):
            _, path = self.dijkstra(waypoints[i], waypoints[i + 1], 100_000, comp_sys)
            if not path:
                p_pen = p_normal
                break
            p_pen.extend(path)

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

    def build_profile(self, start_op: str, end_op: str,
                      via_ops: list[str] | None = None,
                      penalty_m: float = 0.0, comp_sys: list[str] = None) -> pd.DataFrame | None:
        waypoints = [start_op] + (via_ops or []) + [end_op]
        all_steps: list[dict] = []
        for i in range(len(waypoints) - 1):
            _, path = self.dijkstra(waypoints[i], waypoints[i + 1], penalty_m, comp_sys)
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

        ue_gap_ahead: list[float] = []
        for i, step in enumerate(all_steps):
            elec = self.seg_props.get(step["ne_id"], {}).get("electrification", "NONE")
            if comp_sys is not None and elec in comp_sys:
                ue_gap_ahead.append(0.0)
                continue
            gap = 0.0
            j = i
            while j < len(all_steps):
                s = self.seg_props.get(all_steps[j]["ne_id"], {})
                e2 = s.get("electrification", "NONE")
                if comp_sys is not None and e2 in comp_sys:
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

    def search_stations(self, query: str, max_results: int = 80) -> list[tuple[str, str]]:
        q = query.strip().lower()
        if not q: return []
        return [(oid, nm) for nm, oid in self.station_list if q in nm.lower()][:max_results]

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
    def __init__(self, mass_kg, length_m, max_power_kw, aux_power_kw,
                 max_accel, max_decel, traction_type, efficiency,
                 regen_efficiency: float = 0.0,
                 max_speed_kmh: float = 160.0,
                 coast_threshold_m: float = 500.0,
                 comp_sys: list[str] = None):
        self.mass_kg  = mass_kg
        self.eff_mass = mass_kg * 1.08

        # Dynamic Davis coefficients
        m_t = mass_kg / 1000.0
        self.A = 20.0 * m_t
        self.B = 0.2  * m_t
        self.C = 2.5  + 0.03 * length_m

        self.traction   = traction_type
        self.trac_eff   = efficiency if traction_type == "ELECTRIC" else 0.85
        self.regen_eff  = regen_efficiency
        self.aux_w      = aux_power_kw * 1_000.0
        self.max_w      = max_power_kw * 1_000.0
        self.max_accel  = max_accel
        self.max_decel  = max_decel
        self.length_m   = length_m
        self.max_v      = max_speed_kmh / 3.6
        self.coast_threshold_m = coast_threshold_m
        self.comp_sys   = comp_sys

    def _res(self, v: float) -> float:
        return self.A + self.B * v + self.C * v * v

    def run(self, track: TrackProfile, stop_mode: str = "all",
            stop_prob: float = 1.0, dwell: float = 30.0,
            record: bool = False):
        total_m = track.total_km * 1000.0
        g = 9.8186

        stops_km:   list[float] = []
        stop_names: list[str]   = []
        for st in track.stations:
            km = st["km"]
            if km <= 1e-3:
                continue
            if km >= track.total_km - 1e-3:
                continue
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
                                 "gross_kwh", "regen_kwh", "net_kwh", "grad")} if record else None

        # Watchdog limit to prevent infinite loops on impossible climbs
        max_iters = int(total_m / 0.1) + 36000
        iters = 0
        dt = 1.0

        while dist < total_m:
            iters += 1
            if iters > max_iters:
                raise RuntimeError(f"Simulation stalled indefinitely at km {km:.2f} (Speed dropped to 0 and cannot overcome resistance/gradient).")

            km       = dist / 1000.0
            rear_km  = max(0.0, km - self.length_m / 1000.0)
            seg      = track.seg_at(km)
            v_lim    = min(track.v_limit_span(km, rear_km), self.max_v)
            slope    = seg.get("grad", 0.0)
            electr   = seg.get("electrification", "NONE")
            recup_ok = seg.get("recuperation", 0) == 1

            # --- Prune passed stops to prevent desync/infinite stalls ---
            while stops_km and km > stops_km[0] + 0.1:
                stops_km.pop(0)

            coasting = False
            if self.traction == "ELECTRIC" and (self.comp_sys is not None and electr not in self.comp_sys):
                gap_m = track.get_incompatible_gap(km, self.comp_sys)
                if gap_m <= self.coast_threshold_m:
                    coasting = True
                else:
                    raise RuntimeError(
                        f"⚡ Electric vehicle on incompatible track ({electr}) at km {km:.2f} "
                        f"(gap {gap_m / 1000:.1f} km > coasting limit "
                        f"{self.coast_threshold_m / 1000:.1f} km)."
                    )

            f_grad    = self.mass_kg * g * slope
            eff_decel = max(0.05, self.max_decel + g * slope)

            max_safe = v_lim
            next_stop = stops_km[0] if stops_km else None
            if next_stop is not None:
                d2s = (next_stop - km) * 1000.0
                max_safe = min(max_safe, math.sqrt(max(0.0, 2.0 * eff_decel * d2s)))

            if hist is not None:
                vl_front = min(track.v_limit_span(km, km), self.max_v)
                hist["time_s"].append(t_s)
                hist["km"].append(km)
                hist["v_kmh"].append(v * 3.6)
                hist["v_limit_kmh"].append(vl_front * 3.6)
                hist["gross_kwh"].append(e_j / 3_600_000.0)
                hist["regen_kwh"].append(r_j / 3_600_000.0)
                hist["net_kwh"].append((e_j - r_j) / 3_600_000.0)
                hist["grad"].append(slope * 1000.0)

            v_eff  = max(v, 0.5)
            mech_p = reg_p = 0.0

            if v > max_safe + 1e-4:
                f_res    = self._res(v)
                nat_d    = (f_res + f_grad) / self.eff_mass
                brk      = max(0.0, min(self.max_decel, (v - max_safe) - nat_d))
                if recup_ok and not coasting:
                    reg_p = self.eff_mass * brk * v * self.regen_eff
                v = max(0.0, v - (brk + nat_d))

            elif coasting:
                f_res = self._res(v)
                v     = max(0.0, v - (f_res + f_grad) / self.eff_mass)

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
            dist   += v * dt
            t_s    += dt

            if v < 0.5 and any(abs(km - s) <= 0.1 for s in stops_km):
                v = 0.0
                if hist is not None:
                    for pass_no in range(2):
                        vl = min(track.v_limit_span(km, km), self.max_v) * 3.6
                        hist["time_s"].append(t_s)
                        hist["km"].append(km)
                        hist["v_kmh"].append(0.0)
                        hist["v_limit_kmh"].append(vl)
                        hist["gross_kwh"].append(e_j / 3_600_000.0)
                        hist["regen_kwh"].append(r_j / 3_600_000.0)
                        hist["net_kwh"].append((e_j - r_j) / 3_600_000.0)
                        hist["grad"].append(slope * 1000.0)
                        if pass_no == 0:
                            e_j += self.aux_w * dwell
                            t_s += dwell
                else:
                    e_j += self.aux_w * dwell
                    t_s += dwell

                if any(abs(track.total_km - s) <= 0.1 for s in stops_km if abs(km - s) <= 0.1):
                    break

                stops_km = [s for s in stops_km if abs(s - km) > 0.1]

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
    # Graceful fallback handler for cached sessions
    if not isinstance(stats, dict):
        return 0.0
    val = stats.get("net_kwh", 0.0)
    return val / (density * eff) if traction == "DIESEL" else val

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
            xmls = [n for n in zf.namelist() if n.lower().endswith((".xml", ".railml"))]
            if not xmls: return None
            zf.extract(xmls[0], tmp)
            return os.path.join(tmp, xmls[0])
    path = os.path.join(tmp, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.read())
    return path

def _spd_cell_color(v):
    lo, hi = 30.0, 160.0
    t = max(0.0, min(1.0, (float(v) - lo) / (hi - lo)))
    r = int(220 * (1 - t) + 34 * t)
    g = int(38  * (1 - t) + 197 * t)
    b = int(38  * (1 - t) + 94 * t)
    return f"rgba({r},{g},{b},0.25)"

def make_profile_chart(df: pd.DataFrame) -> go.Figure:
    total_km = float(df["cum_km"].max())

    mid_km, seg_widths = [], []
    for i in range(len(df) - 1):
        x0 = float(df["cum_km"].iloc[i])
        x1 = float(df["cum_km"].iloc[i + 1])
        mid_km.append((x0 + x1) / 2)
        seg_widths.append(max(x1 - x0, 0.001))

    stops_x = df[df["stop_type"] == "X"]
    stops_r = df[df["stop_type"] == "R"]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Statutory Track Speed Limit [km/h]",
                        "Gradient [‰]  (↑ uphill  ↓ downhill)",
                        "Electrification"),
        row_heights=[0.48, 0.32, 0.20], vertical_spacing=0.055,
    )

    # ── Speed — STEPPED line
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

    n_x = max(len(stops_x), 1)
    label_every = max(1, n_x // 20)
    labelled_x  = stops_x.iloc[::label_every]
    unlabelled_x = stops_x[~stops_x.index.isin(labelled_x.index)]

    if not labelled_x.empty:
        fig.add_trace(go.Scatter(
            x=labelled_x["cum_km"], y=labelled_x["speed_kmh"],
            mode="markers+text",
            marker=dict(symbol="diamond", size=9, color=C["accent"], line=dict(color="white", width=1.5)),
            text=labelled_x["station_name"], textposition="top center", textfont=dict(size=7, color=C["dark"]),
            name="Station (X)", hovertemplate="<b>%{text}</b><br>km %{x:.2f}  %{y:.0f} km/h<extra></extra>",
        ), row=1, col=1)

    if not unlabelled_x.empty:
        fig.add_trace(go.Scatter(
            x=unlabelled_x["cum_km"], y=unlabelled_x["speed_kmh"],
            mode="markers",
            marker=dict(symbol="diamond", size=6, color=C["accent"], line=dict(color="white", width=1)),
            showlegend=False, text=unlabelled_x["station_name"],
            hovertemplate="<b>%{text}</b><br>km %{x:.2f}  %{y:.0f} km/h<extra></extra>",
        ), row=1, col=1)

    if not stops_r.empty:
        fig.add_trace(go.Scatter(
            x=stops_r["cum_km"], y=stops_r["speed_kmh"],
            mode="markers",
            marker=dict(symbol="circle", size=7, color=C["yellow"], line=dict(color="white", width=1)),
            name="Halt (R)", text=stops_r["station_name"],
            hovertemplate="<b>%{text}</b><br>km %{x:.2f}  %{y:.0f} km/h<extra></extra>",
        ), row=1, col=1)

    # ── Gradient — Solid Bar Fill to eliminate Plotly 0-crossover bugs
    grads = df["gradient_perm"].iloc[:-1] if len(df) > 1 else df["gradient_perm"]
    pos_g = grads.clip(lower=0)
    neg_g = grads.clip(upper=0)

    fig.add_trace(go.Bar(
        x=mid_km, y=pos_g, width=seg_widths, marker_color=C["red"], name="Uphill",
        hovertemplate="km %{x:.2f}<br>+%{y:.1f} ‰<extra></extra>"
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        x=mid_km, y=neg_g, width=seg_widths, marker_color=C["primary"], name="Downhill",
        hovertemplate="km %{x:.2f}<br>%{y:.1f} ‰<extra></extra>"
    ), row=2, col=1)

    # ── Electrification band
    elec = df["electrification"].iloc[:-1] if len(df) > 1 else df["electrification"]
    seg_colors = [elec_color(e) for e in elec]
    seg_hover = list(elec)

    fig.add_trace(go.Bar(
        x=mid_km, y=[1] * len(mid_km), width=seg_widths,
        marker_color=seg_colors, name="Electrification", hovertext=seg_hover,
        hovertemplate="%{hovertext}<br>km %{x:.2f}<extra></extra>", showlegend=False,
    ), row=3, col=1)

    for _, sr in labelled_x.iterrows():
        fig.add_vline(x=sr["cum_km"], line_width=0.7, line_dash="dot", line_color="#CBD5E1", row=1, col=1)

    spd_min = float(df["speed_kmh"].min())
    spd_max = float(df["speed_kmh"].max())
    spd_lo  = max(0, spd_min * 0.75)
    spd_hi  = spd_max * 1.25

    fig.update_xaxes(title_text="Distance from departure [km]", row=3, col=1, gridcolor=C["light"])
    fig.update_yaxes(title_text="Speed [km/h]", row=1, col=1, gridcolor=C["light"], range=[spd_lo, spd_hi])
    fig.update_yaxes(title_text="Gradient [‰]", row=2, col=1, gridcolor=C["light"], zeroline=True, zerolinecolor="#CBD5E1")
    fig.update_yaxes(showticklabels=False, row=3, col=1, range=[0, 1.1])

    # Adjusted Height, Margin, and Legend position to fix subtitle overlap issues
    fig.update_layout(
        height=800, barmode="overlay", paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0, bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1),
        margin=dict(t=100, b=10, l=60, r=20),
        font=dict(family="Inter, sans-serif", size=12),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=C["light"], showgrid=True)
    return fig

def make_route_map(df: pd.DataFrame, via_ops: list, parser: DYPODParser, show_halts: bool = True) -> go.Figure:
    map_df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    fig    = go.Figure()

    if not map_df.empty:
        runs: list[dict] = []
        cur_electr  = None
        cur_lats:  list = []
        cur_lons:  list = []
        cur_hover: list = []

        for i in range(len(map_df) - 1):
            row_a = map_df.iloc[i]
            row_b = map_df.iloc[i + 1]
            seg_electr = str(row_a["electrification"])

            if seg_electr != cur_electr:
                if cur_lats:
                    runs.append(dict(electr=cur_electr, lats=cur_lats, lons=cur_lons, hover=cur_hover))
                cur_electr = seg_electr
                cur_lats   = [float(row_a["lat"])]
                cur_lons   = [float(row_a["lon"])]
                cur_hover  = [f"{seg_electr}<br>km {row_a['cum_km']:.2f}"]

            cur_lats.append(float(row_b["lat"]))
            cur_lons.append(float(row_b["lon"]))
            cur_hover.append(f"{seg_electr}<br>km {row_b['cum_km']:.2f}")

        if cur_lats:
            runs.append(dict(electr=cur_electr, lats=cur_lats, lons=cur_lons, hover=cur_hover))

        seen_labels: set = set()
        for run in runs:
            show_leg = run["electr"] not in seen_labels
            seen_labels.add(run["electr"])
            col = elec_color(run["electr"])
            lw  = 5 if run["electr"] != "NONE" else 4
            fig.add_trace(go.Scattermap(
                lat=run["lats"], lon=run["lons"], mode="lines",
                line=dict(width=lw, color=col), name=run["electr"],
                showlegend=show_leg, hovertext=run["hover"], hovertemplate="%{hovertext}<extra></extra>",
            ))

    if show_halts:
        r_pts = map_df[map_df["stop_type"] == "R"]
        if not r_pts.empty:
            fig.add_trace(go.Scattermap(
                lat=r_pts["lat"].tolist(), lon=r_pts["lon"].tolist(), mode="markers",
                marker=dict(size=7, color=C["yellow"]), name="Halt (R)",
                customdata=r_pts["station_name"].values, hovertemplate="<b>%{customdata}</b><extra></extra>",
            ))

    x_pts = map_df[map_df["stop_type"] == "X"]
    if not x_pts.empty:
        fig.add_trace(go.Scattermap(
            lat=x_pts["lat"].tolist(), lon=x_pts["lon"].tolist(), mode="markers+text",
            marker=dict(size=11, color=C["accent"]), text=x_pts["station_name"].tolist(), textposition="top right",
            textfont=dict(size=9, color=C["dark"]), name="Station (X)",
            customdata=x_pts[["cum_km", "speed_kmh", "gradient_perm"]].values,
            hovertemplate=("<b>%{text}</b><br>km %{customdata[0]:.2f}<br>%{customdata[1]:.0f} km/h<br>grad %{customdata[2]:.1f} ‰<extra></extra>"),
        ))

    for vid in (via_ops or []):
        info = parser.op_info.get(vid, {})
        if info.get("lat") and info.get("lon"):
            fig.add_trace(go.Scattermap(
                lat=[info["lat"]], lon=[info["lon"]], mode="markers+text",
                marker=dict(size=18, color=C["yellow"]), text=[info["name"]], textposition="top right",
                textfont=dict(size=10, color="#92400E"), name=f"Via: {info['name']}", showlegend=False,
            ))

    if not map_df.empty:
        for row_lat, row_lon, row_name, tpos, col, leg in [
            (float(map_df["lat"].iloc[0]),  float(map_df["lon"].iloc[0]), map_df["station_name"].iloc[0],  "top right", C["green"], "Departure"),
            (float(map_df["lat"].iloc[-1]), float(map_df["lon"].iloc[-1]), map_df["station_name"].iloc[-1], "top left",  C["red"],   "Arrival"),
        ]:
            fig.add_trace(go.Scattermap(
                lat=[row_lat], lon=[row_lon], mode="markers+text", marker=dict(size=20, color=col),
                text=[row_name], textposition=tpos, textfont=dict(size=11, color=C["dark"], family="Inter, sans-serif"),
                name=leg, hovertemplate="<b>%{text}</b><extra></extra>",
            ))

    lat_c = float(map_df["lat"].mean()) if not map_df.empty else 50.0
    lon_c = float(map_df["lon"].mean()) if not map_df.empty else 15.5
    fig.update_layout(
        map=dict(style="open-street-map", center=dict(lat=lat_c, lon=lon_c), zoom=7),
        height=540, margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor="#E2E8F0", borderwidth=1, x=0.01, y=0.99, font=dict(size=11)),
    )
    return fig


def make_kinematic_charts(hist: dict, stop_names: list[str],
                          df_profile: pd.DataFrame,
                          x_axis: str = "Distance (km)") -> tuple[go.Figure, go.Figure, go.Figure]:
    base = pd.to_datetime("1970-01-01")
    time_dt = [base + pd.to_timedelta(s, unit="s") for s in hist["time_s"]]
    dist_km  = hist["km"]

    use_time = x_axis == "Time (MM:SS)"
    x_data   = time_dt if use_time else dist_km
    dur_s    = hist["time_s"][-1]
    if use_time:
        buf = pd.Timedelta(seconds=max(30, int(dur_s * 0.02)))
        x_min = time_dt[0]  - buf; x_max = time_dt[-1] + buf
        x_fmt = "%H:%M:%S" if dur_s >= 3600 else "%M:%S"
        x_title = "Elapsed time (hh:mm:ss)" if dur_s >= 3600 else "Elapsed time (mm:ss)"
    else:
        buf_km = max(0.5, (dist_km[-1] - dist_km[0]) * 0.02)
        x_min = dist_km[0] - buf_km; x_max = dist_km[-1] + buf_km
        x_fmt = None; x_title = "Cumulative distance [km]"

    stops_set = set(stop_names)

    # Force inclusion of start and end stations to ensure they are fully visible
    # on the axis regardless of stop filtering logic
    all_stops_df = pd.DataFrame()
    if df_profile is not None and not df_profile.empty:
        df_plot = df_profile.copy()

        # Robustly find the first and last actual stations in this specific leg
        valid_stations = df_plot[df_plot["station_name"].str.strip() != ""]
        if not valid_stations.empty:
            first_idx = valid_stations.index[0]
            last_idx = valid_stations.index[-1]
            df_plot.loc[first_idx, "stop_type"] = "X"
            df_plot.loc[last_idx, "stop_type"] = "X"
            stops_set.add(df_plot.loc[first_idx, "station_name"])
            stops_set.add(df_plot.loc[last_idx, "station_name"])

        all_stops_df = df_plot[df_plot["stop_type"].isin(["X","R"])]

    v_max = max(max(hist["v_limit_kmh"]), max(hist["v_kmh"])) * 1.05 if hist["v_kmh"] else 10
    km_arr = np.array(hist["km"])

    shapes_speed = []
    annotations_speed = []
    shapes_grad = []
    shapes_energy = []
    annotations_energy = []

    for _, srow in all_stops_df.iterrows():
        skm   = float(srow["cum_km"])
        stype = srow["stop_type"]
        sname = srow["station_name"]
        if not (km_arr.min()-0.1 <= skm <= km_arr.max()+0.1): continue
        color = C["grey"] if stype == "X" else (C["primary"] if sname in stops_set else C["red"])
        idx   = int(np.argmin(np.abs(km_arr - skm)))
        x_pos = time_dt[idx] if use_time else float(dist_km[idx])

        line_dict = dict(type="line", xref="x", yref="paper", x0=x_pos, x1=x_pos,
                         y0=0, y1=1, line=dict(color=color, width=1, dash="dot"), layer="below")

        shapes_speed.append(line_dict)
        shapes_grad.append(line_dict)
        shapes_energy.append(line_dict)

        annot_dict = dict(x=x_pos, y=-0.20, xref="x", yref="paper",
                          text=sname, showarrow=False, font=dict(size=10, color=color),
                          xanchor="right", yanchor="top", textangle=-45)

        annotations_speed.append(annot_dict)
        annotations_energy.append(annot_dict)


    # 1. SPEED CHART
    fig_speed = go.Figure()
    fig_speed.add_trace(go.Scatter(x=x_data, y=hist["v_limit_kmh"], name="Speed Limit",
        line=dict(color=C["red"], dash="dot", width=1.8, shape="hv")))
    fig_speed.add_trace(go.Scatter(x=x_data, y=hist["v_kmh"], name="Actual Speed",
        line=dict(color=C["primary"], width=2.5),
        fill="tozeroy", fillcolor=C["bg_blue"]))

    fig_speed.update_layout(
        height=380, margin=dict(l=60, r=40, t=20, b=120), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1),
        shapes=shapes_speed, annotations=annotations_speed,
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
        xaxis=dict(title=x_title, range=[x_min, x_max], tickformat=x_fmt, showgrid=True, gridcolor=C["light"], showticklabels=True),
        yaxis=dict(title_text="Speed [km/h]", showgrid=True, gridcolor=C["light"])
    )


    # 2. GRADIENT CHART
    fig_grad = go.Figure()
    if "grad" in hist:
        g_arr = np.array(hist["grad"])
        pos_g = np.clip(g_arr, 0, None)
        neg_g = np.clip(g_arr, None, 0)

        fig_grad.add_trace(go.Scatter(x=x_data, y=pos_g, name="Uphill",
            line=dict(color=C["red"], width=0, shape="hv"),
            fill="tozeroy", fillcolor="rgba(220,38,38,0.55)", showlegend=False))
        fig_grad.add_trace(go.Scatter(x=x_data, y=neg_g, name="Downhill",
            line=dict(color=C["primary"], width=0, shape="hv"),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.55)", showlegend=False))

    fig_grad.update_layout(
        height=250, margin=dict(l=60, r=40, t=20, b=60), hovermode="x unified",
        shapes=shapes_grad,
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
        xaxis=dict(title=x_title, range=[x_min, x_max], tickformat=x_fmt, showgrid=True, gridcolor=C["light"], showticklabels=True),
        yaxis=dict(title_text="Gradient [‰]", showgrid=True, gridcolor=C["light"], zeroline=True, zerolinecolor="#CBD5E1")
    )


    # 3. ENERGY CHART
    fig_energy = go.Figure()
    fig_energy.add_trace(go.Scatter(x=x_data, y=hist["gross_kwh"], name="Gross Energy",
        line=dict(color=C["yellow"], width=2),
        fill="tozeroy", fillcolor="rgba(202,138,4,0.15)")) # Light overlay for gross energy
    fig_energy.add_trace(go.Scatter(x=x_data, y=hist["regen_kwh"], name="Recuperated",
        line=dict(color=C["green"], width=2, dash="dash")))
    fig_energy.add_trace(go.Scatter(x=x_data, y=hist["net_kwh"], name="Net Energy",
        line=dict(color=C["secondary"], width=2.5)))

    fig_energy.update_layout(
        height=380, margin=dict(l=60, r=40, t=20, b=120), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1),
        shapes=shapes_energy, annotations=annotations_energy,
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
        xaxis=dict(title=x_title, range=[x_min, x_max], tickformat=x_fmt, showgrid=True, gridcolor=C["light"], showticklabels=True),
        yaxis=dict(title_text="Energy [kWh]", showgrid=True, gridcolor=C["light"])
    )

    return fig_speed, fig_grad, fig_energy


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
.block-container{padding-top:2.5rem}
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
    op_start=None, op_end=None, comp_sys=None, combined_vias=[],
    rebuild_profile=False, scenario_mode="Single Journey",
    last_uploaded_id=None, scenario_terminals=[]
)
for k, v in _SS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚆 DYPOD Simulator")
    st.caption("Czech railway track profile & energy analysis")
    st.markdown("---")

    st.markdown('<div class="sec">📂 Load railML</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload railML or zip", type=["xml", "railml", "zip"],
        label_visibility="collapsed",
    )
    xml_path = None
    if uploaded:
        file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("last_uploaded_id") != file_id:
            st.session_state.last_uploaded_id = file_id
            xp = load_xml_from_upload(uploaded)
            for k, v in _SS_DEFAULTS.items():
                st.session_state[k] = v
            st.session_state.xml_path = xp
            st.session_state.last_uploaded_id = file_id
        xml_path = st.session_state.xml_path
    else:
        local = sorted(
            glob.glob("*.xml") + glob.glob("*.railml") + glob.glob("*.zip") +
            glob.glob("/tmp/*.xml") + glob.glob("/tmp/*.railml")
        )
        if local:
            sel = st.selectbox("Or pick local file", local, label_visibility="collapsed")
            if sel != st.session_state.xml_path:
                for k, v in _SS_DEFAULTS.items():
                    st.session_state[k] = v
                st.session_state.xml_path = sel
                st.session_state.last_uploaded_id = None
            xml_path = sel

    @st.cache_resource(show_spinner="Parsing railML infrastructure…")
    def _load(path: str) -> DYPODParser:
        if path.lower().endswith(".zip"):
            tmp = tempfile.mkdtemp()
            with zipfile.ZipFile(path) as zf:
                xmls = [n for n in zf.namelist() if n.lower().endswith((".xml", ".railml"))]
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

    st.markdown('<div class="sec">🗺️ Route</div>', unsafe_allow_html=True)

    def _station_selector(label: str, key_q: str, key_sel: str) -> str | None:
        query = st.text_input(label, key=key_q, placeholder="Type to search…")
        results = parser.search_stations(query)
        if not results:
            if query.strip():
                st.caption("⚠️ No matching stations")
            return None
        options = {f"{nm}": oid for oid, nm in results}
        chosen = st.selectbox(
            f"↳ {label}", options=list(options.keys()),
            key=key_sel, label_visibility="collapsed", index=None
        )
        return options.get(chosen) if chosen else None

    scenario_mode = st.radio("Routing Mode", ["Single Journey", "Round-Trip / Shift", "Custom Itinerary"])
    st.session_state.scenario_mode = scenario_mode

    scenario_vias = []
    station_dict = {nm: oid for nm, oid in parser.station_list}
    station_names = list(station_dict.keys())

    if station_names:
        if scenario_mode == "Single Journey":
            op_start_name = st.selectbox("🔍 Departure station", station_names, index=None)
            op_end_name   = st.selectbox("🔍 Arrival station", station_names, index=None)
            op_start = station_dict.get(op_start_name) if op_start_name else None
            op_end   = station_dict.get(op_end_name) if op_end_name else None

        elif scenario_mode == "Round-Trip / Shift":
            op_start_name = st.selectbox("🔍 Terminal A", station_names, index=None)
            op_end_name   = st.selectbox("🔍 Terminal B", station_names, index=None)
            legs = st.number_input("Number of legs (1 direction = 1 leg)", min_value=2, max_value=20, value=2)
            op_a = station_dict.get(op_start_name) if op_start_name else None
            op_b = station_dict.get(op_end_name) if op_end_name else None
            if op_a and op_b:
                seq = [op_a if i % 2 == 0 else op_b for i in range(legs + 1)]
                op_start = seq[0]
                op_end = seq[-1]
                scenario_vias = seq[1:-1]
            else:
                op_start, op_end = None, None

        elif scenario_mode == "Custom Itinerary":
            legs = st.number_input("Number of consecutive legs", min_value=2, max_value=15, value=2)
            seq_names = []
            for i in range(int(legs) + 1):
                if i == 0:
                    name = st.selectbox("Start Station", station_names, index=None, key=f"cust_{i}")
                else:
                    name = st.selectbox(f"End of Leg {i}", station_names, index=None, key=f"cust_{i}")
                seq_names.append(name)

            seq = [station_dict.get(nm) for nm in seq_names if nm]
            if len(seq) == int(legs) + 1:
                op_start = seq[0]
                op_end = seq[-1]
                scenario_vias = seq[1:-1]
            else:
                op_start, op_end = None, None
    else:
        op_start, op_end = None, None
        st.warning("No passenger stations found in the data.")

    is_valid_route = bool(op_start and op_end and (op_start != op_end or scenario_vias))

    st.markdown('<div class="sec">🚃 Vehicle</div>', unsafe_allow_html=True)
    veh = st.selectbox("Preset", ["Custom"] + list(PREDEFINED_VEHICLES.keys()), label_visibility="collapsed")
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
        max_speed  = st.number_input("Max speed (km/h)", value=160, min_value=30, max_value=350)
    else:
        p = PREDEFINED_VEHICLES[veh]
        traction, mass, length = p["traction"], p["mass"], p["length"]
        power, aux_power       = p["power"], p["aux_power"]
        accel, decel, efficiency = p["accel"], p["decel"], p["efficiency"]
        max_speed              = p["max_speed"]

    unit_lbl = "L fuel" if traction == "DIESEL" else "kWh"

    # Global vehicle tweaks
    if traction == "DIESEL":
        diesel_d = st.slider("Diesel Energy Density (kWh/L)", 8.0, 12.0, 10.0, 0.1)
        regen_efficiency = 0.0
    else:
        diesel_d = 10.0
        regen_eff_default = p.get("regen_eff", 0.75) if veh != "Custom" else 0.75
        regen_efficiency = st.slider("Recuperation Efficiency (%)", 0, 100, int(regen_eff_default * 100)) / 100.0

    if traction == "ELECTRIC":
        if veh == "Custom":
            comp_sys = st.multiselect("Compatible catenary systems", ["3,000V/0Hz", "25,000V/50Hz", "1,500V/0Hz", "15,000V/16.7Hz"], default=["3,000V/0Hz", "25,000V/50Hz"])
        else:
            comp_sys = st.multiselect("Compatible catenary systems", ["3,000V/0Hz", "25,000V/50Hz", "1,500V/0Hz", "15,000V/16.7Hz"], default=p.get("systems", []))

        st.markdown("**⚡ Compatible route preference**")
        elec_reroute = st.toggle("Prefer compatible track", value=False, help="Penalises incompatible/non-electrified segments in route search.")
        pen_km = st.slider("Detour tolerance (km per incompatible segment)", 10, 500, 100, 10) if elec_reroute else 0
    else:
        comp_sys = None
        elec_reroute = False
        pen_km = 0

    btn_profile = st.button("🗺️  Build Track Profile", use_container_width=True, type="primary", disabled=not is_valid_route)

    st.markdown('<div class="sec">⚙️ Simulation</div>', unsafe_allow_html=True)
    dwell     = st.number_input("Station dwell (s)", value=30, step=5)
    stop_mode = st.radio("Stop mode", options=["all", "random", "none"],
                         format_func={"all": "All stops", "random": "Stochastic", "none": "Express"}.get, horizontal=True)
    stop_prob = 1.0
    if stop_mode == "random":
        stop_prob = st.slider("Request-stop probability", 0.0, 1.0, 0.5, 0.05)

    coast_threshold_m = 500.0
    if traction == "ELECTRIC":
        st.markdown("**⚡ Electric coasting**")
        coast_km = st.slider("Max coastable incompatible gap (km)", 0.1, 10.0, 0.5, 0.1)
        coast_threshold_m = coast_km * 1000.0

    st.markdown('<div class="sec">🎲 Monte Carlo</div>', unsafe_allow_html=True)
    mc_n     = st.number_input("Runs per probability", 20, 500, 100, 10)
    mc_probs = st.multiselect("Probabilities to sweep", options=[1.0, 0.8, 0.6, 0.4, 0.2, 0.0], default=[1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

    st.markdown('<div class="sec">📊 Display Options</div>', unsafe_allow_html=True)
    x_axis_choice = st.radio("Kinematic X-axis", ["Distance (km)", "Time (MM:SS)"])

    cb1, cb2 = st.columns(2)
    btn_run = cb1.button("▶️ Run",  use_container_width=True, disabled=st.session_state.profile_df is None)
    btn_mc  = cb2.button("🎲 MC",   use_container_width=True, disabled=st.session_state.profile_df is None)


# ─── Actions ─────────────────────────────────────────────────────────────────
if (btn_profile or st.session_state.rebuild_profile) and is_valid_route:
    st.session_state.rebuild_profile = False
    pen = pen_km * 1000.0 if elec_reroute else 0.0
    combined_vias = st.session_state.via_ops + scenario_vias
    scenario_terminals = [op_start] + scenario_vias + [op_end]

    with st.spinner("Finding path and building continuous profile…"):
        ea  = parser.analyse_electrification(op_start, op_end, via_ops=combined_vias, comp_sys=comp_sys)
        df  = parser.build_profile(op_start, op_end,
                                   via_ops=combined_vias,
                                   penalty_m=pen, comp_sys=comp_sys)
    if df is None or df.empty:
        st.error("No path found between the selected sequence of stations.")
    else:
        st.session_state.update(profile_df=df, op_start=op_start, op_end=op_end,
                                elec_analysis=ea, rep_result=None, mc_result=None,
                                comp_sys=comp_sys, combined_vias=combined_vias,
                                scenario_terminals=scenario_terminals)

if btn_run and st.session_state.profile_df is not None:
    with st.spinner("Running kinematic physics simulation…"):
        try:
            df_full = st.session_state.profile_df
            terminals = st.session_state.get("scenario_terminals", [])

            terminal_kms = [0.0]
            if len(terminals) > 1:
                current_term_idx = 1
                for idx, row in df_full.iterrows():
                    if current_term_idx < len(terminals):
                        if row["op_id"] == terminals[current_term_idx] and row["cum_km"] > terminal_kms[-1] + 0.1:
                            terminal_kms.append(row["cum_km"])
                            current_term_idx += 1

            if len(terminal_kms) != len(terminals):
                terminal_kms = [0.0, df_full["cum_km"].max()]

            sim   = TrainSimulator(mass, length, power, aux_power,
                                   accel, decel, traction, efficiency,
                                   max_speed_kmh=max_speed,
                                   regen_efficiency=regen_efficiency,
                                   coast_threshold_m=coast_threshold_m,
                                   comp_sys=st.session_state.comp_sys)

            leg_results = []
            total_gross, total_regen, total_net, total_net_worst, total_time = 0.0, 0.0, 0.0, 0.0, 0.0

            for i in range(len(terminal_kms) - 1):
                start_km = terminal_kms[i]
                end_km = terminal_kms[i+1]

                df_leg = df_full[(df_full["cum_km"] >= start_km - 1e-4) & (df_full["cum_km"] <= end_km + 1e-4)].copy()
                df_leg["cum_km"] -= start_km

                track_leg = TrackProfile(df_leg)
                total_leg_stops = len(track_leg.stations)

                hist_leg, snames_leg, stats_leg = sim.run(track_leg, stop_mode=stop_mode, stop_prob=stop_prob, dwell=dwell, record=True)
                _, _, stats_worst_leg = sim.run(track_leg, stop_mode="all", stop_prob=1.0, dwell=dwell, record=False)

                leg_results.append({
                    "leg_num": i + 1,
                    "start_name": parser.op_name(terminals[i]) if len(terminals) > 1 else "",
                    "end_name": parser.op_name(terminals[i+1]) if len(terminals) > 1 else "",
                    "df_leg": df_leg,
                    "hist": hist_leg,
                    "stats": stats_leg,
                    "stats_worst": stats_worst_leg,
                    "snames": snames_leg,
                    "total_possible_stops": total_leg_stops
                })

                total_gross += stats_leg["gross_kwh"]
                total_regen += stats_leg["regen_kwh"]
                total_net += stats_leg["net_kwh"]
                total_net_worst += stats_worst_leg["net_kwh"]
                total_time += stats_leg["journey_time_s"]

            st.session_state.rep_result = dict(
                leg_results=leg_results,
                total_stats={"gross_kwh": total_gross, "regen_kwh": total_regen, "net_kwh": total_net, "journey_time_s": total_time},
                total_net_worst=total_net_worst,
                traction=traction, diesel_d=diesel_d, efficiency=efficiency,
                unit_lbl=unit_lbl, vehicle_name=veh, dwell=dwell,
                coast_threshold_m=coast_threshold_m,
            )
        except RuntimeError as e:
            st.error(str(e))

if btn_mc and st.session_state.profile_df is not None and mc_probs:
    pb = None
    with st.spinner(f"Monte Carlo — {int(mc_n)} × {len(mc_probs)} probabilities…"):
        try:
            df_full = st.session_state.profile_df
            terminals = st.session_state.get("scenario_terminals", [])

            terminal_kms = [0.0]
            if len(terminals) > 1:
                current_term_idx = 1
                for idx, row in df_full.iterrows():
                    if current_term_idx < len(terminals):
                        if row["op_id"] == terminals[current_term_idx] and row["cum_km"] > terminal_kms[-1] + 0.1:
                            terminal_kms.append(row["cum_km"])
                            current_term_idx += 1

            if len(terminal_kms) != len(terminals):
                terminal_kms = [0.0, df_full["cum_km"].max()]

            leg_tracks = []
            for i in range(len(terminal_kms) - 1):
                start_km = terminal_kms[i]
                end_km = terminal_kms[i+1]
                df_leg = df_full[(df_full["cum_km"] >= start_km - 1e-4) & (df_full["cum_km"] <= end_km + 1e-4)].copy()
                df_leg["cum_km"] -= start_km
                leg_tracks.append(TrackProfile(df_leg))

            sim   = TrainSimulator(mass, length, power, aux_power,
                                   accel, decel, traction, efficiency,
                                   max_speed_kmh=max_speed,
                                   regen_efficiency=regen_efficiency,
                                   coast_threshold_m=coast_threshold_m,
                                   comp_sys=st.session_state.comp_sys)

            rows_mc_overall = []
            rows_mc_legs = {i: [] for i in range(len(leg_tracks))}

            pb = st.progress(0.0, text="Initializing Monte Carlo Engine...")
            total_runs = len(mc_probs) * int(mc_n)
            completed_runs = 0

            for p_val in sorted(mc_probs, reverse=True):
                sm = ("all" if p_val == 1.0 else "none" if p_val == 0.0 else "random")

                overall_e_list, overall_t_list = [], []
                leg_e_lists = {i: [] for i in range(len(leg_tracks))}
                leg_t_lists = {i: [] for i in range(len(leg_tracks))}

                for _ in range(int(mc_n)):
                    run_e = 0.0
                    run_t = 0.0
                    for i, track_leg in enumerate(leg_tracks):
                        _, _, st_ = sim.run(track_leg, sm, p_val, dwell)
                        e_val = to_unit(st_, traction, diesel_d, efficiency)
                        t_val = st_["journey_time_s"]

                        leg_e_lists[i].append(e_val)
                        leg_t_lists[i].append(t_val)
                        run_e += e_val
                        run_t += t_val

                    overall_e_list.append(run_e)
                    overall_t_list.append(run_t)

                    completed_runs += 1
                    if completed_runs % max(1, total_runs // 20) == 0:
                        if pb is not None: pb.progress(completed_runs / total_runs, text=f"Computing permutations ({completed_runs}/{total_runs})")

                rows_mc_overall.append(dict(
                    prob=f"{int(p_val * 100)}%", p_num=p_val,
                    mean_e=np.mean(overall_e_list),  std_e=np.std(overall_e_list),
                    min_e=np.min(overall_e_list),    max_e=np.max(overall_e_list),
                    mean_t=np.mean(overall_t_list),  min_t=np.min(overall_t_list), max_t=np.max(overall_t_list),
                ))

                for i in range(len(leg_tracks)):
                    rows_mc_legs[i].append(dict(
                        prob=f"{int(p_val * 100)}%", p_num=p_val,
                        mean_e=np.mean(leg_e_lists[i]),  std_e=np.std(leg_e_lists[i]),
                        min_e=np.min(leg_e_lists[i]),    max_e=np.max(leg_e_lists[i]),
                        mean_t=np.mean(leg_t_lists[i]),  min_t=np.min(leg_t_lists[i]), max_t=np.max(leg_t_lists[i]),
                    ))

            if pb is not None: pb.empty()

            mc_df_overall = pd.DataFrame(rows_mc_overall)
            mc_df_overall["savings"] = (mc_df_overall["mean_e"].max() - mc_df_overall["mean_e"]).clip(lower=0)
            mc_df_overall["unit"] = unit_lbl

            legs_results = []
            for i in range(len(leg_tracks)):
                df_l = pd.DataFrame(rows_mc_legs[i])
                df_l["savings"] = (df_l["mean_e"].max() - df_l["mean_e"]).clip(lower=0)
                df_l["unit"] = unit_lbl

                s_name = leg_tracks[i].df["station_name"].iloc[0] if not leg_tracks[i].df.empty else ""
                e_name = leg_tracks[i].df["station_name"].iloc[-1] if not leg_tracks[i].df.empty else ""

                legs_results.append({
                    "leg_num": i + 1,
                    "start_name": s_name,
                    "end_name": e_name,
                    "df": df_l
                })

            st.session_state.mc_result = {
                "overall": mc_df_overall,
                "legs": legs_results
            }

        except RuntimeError as e:
            if pb is not None: pb.empty()
            st.error(str(e))

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_prof, tab_edit, tab_run_t, tab_mc_t = st.tabs([
    "🗺️ Infrastructure Limits", "✏️ Profile Editor", "▶️ Kinematic Simulation", "🎲 Monte Carlo",
])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – TRACK PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_prof:
    df  = st.session_state.profile_df
    ea  = st.session_state.elec_analysis
    cs  = st.session_state.comp_sys

    if df is None:
        st.markdown("### 👈 Configure your Route/Scenario in the sidebar and click **Build Track Profile**")
        st.info("Stitches meso-level railML segments into a direction-aware profile: "
                "stepped speed limits, gradient, electrification, GPS coordinates. Handles continuous multi-leg shifts.")
        st.stop()

    total_km = float(df["cum_km"].max())
    stops_x  = df[df["stop_type"] == "X"]
    stops_r  = df[df["stop_type"] == "R"]

    incomp_km = sum(s.get("length_m", 0) for s in df.to_dict('records') if cs is not None and s.get("electrification") not in cs) / 1000
    comp_km   = total_km - incomp_km
    sn = df["station_name"].iloc[0]
    en = df["station_name"].iloc[-1]

    st.markdown(f"### 📍 {sn}  →  {en}")

    # Electrification alerts
    if ea is not None and traction == "ELECTRIC":
        ue = ea["incomp_km"]; gw = ea["gateway_km"]
        coast_lbl = f"{coast_threshold_m/1000:.1f} km"
        if ue == 0:
            st.markdown('<div class="ok-box">✅ <b>Fully compatible electrified route.</b></div>', unsafe_allow_html=True)
        elif ue <= coast_threshold_m/1000:
            st.markdown(f'<div class="ok-box">⚡ Short incompatible gap: <b>{ue:.2f} km</b> — within coasting limit ({coast_lbl}). Vehicle will coast on inertia.</div>', unsafe_allow_html=True)
        elif abs(gw - ue) < 0.1 and gw > 0:
            st.markdown(f'<div class="warn-box">⚠️ Unavoidable run-in: <b>{gw:.1f} km</b> incompatible track. No compatible exit from <b>{sn}</b>. Raise coasting limit or use diesel.</div>', unsafe_allow_html=True)
        elif ea.get("has_alternative") and not elec_reroute:
            st.markdown(f'<div class="warn-box">⚡ <b>{ue:.1f} km incompatible track.</b> Enable <b>Prefer compatible track</b> to save <b>{ea["saving_km"]:.1f} km</b> (+{ea["detour_km"]:.0f} km detour). Or raise coasting limit ({coast_lbl}).</div>', unsafe_allow_html=True)
        elif elec_reroute and ea.get("has_alternative"):
            st.markdown(f'<div class="ok-box">✅ <b>Compatible routing active.</b> Saved <b>{ea["saving_km"]:.1f} km</b> incompatible track (+{ea["detour_km"]:.0f} km detour). Remaining: <b>{ea["penalised_incomp_km"]:.1f} km</b> unavoidable (coast limit: {coast_lbl}).</div>', unsafe_allow_html=True)
        elif ue > 0:
            st.markdown(f'<div class="danger-box">🚫 <b>{ue:.1f} km incompatible track</b> exceeds coasting limit ({coast_lbl}). Raise limit, use diesel, or choose different stations.</div>', unsafe_allow_html=True)

    # KPI row
    kc = st.columns(6)
    kc[0].markdown(kpi_card(f"{total_km:.1f} km",       "Route length"),            unsafe_allow_html=True)
    kc[1].markdown(kpi_card(str(len(stops_x)),           "Mandatory stops (X)"),    unsafe_allow_html=True)
    kc[2].markdown(kpi_card(str(len(stops_r)),           "Request halts (R)"),      unsafe_allow_html=True)
    kc[3].markdown(kpi_card(f"{df['speed_kmh'].max():.0f} km/h", "Max speed"),      unsafe_allow_html=True)
    kc[4].markdown(kpi_card(f"{df['gradient_perm'].abs().max():.0f} ‰", "Max gradient"), unsafe_allow_html=True)

    if traction == "ELECTRIC":
        kc[5].markdown(kpi_card(f"{comp_km:.0f} km",f"Compatible ({100*comp_km/total_km if total_km>0 else 0:.0f}%)",
                                 delta=(f"⚠️ {incomp_km:.0f} km incompatible" if incomp_km>0 else "✅ Fully compatible"),
                                 color=(C["red"] if incomp_km>0 else C["green"])), unsafe_allow_html=True)
    else:
        kc[5].markdown(kpi_card(f"{total_km:.0f} km", "Diesel mode", color=C["green"]), unsafe_allow_html=True)

    st.write("")

    st.plotly_chart(make_profile_chart(df), use_container_width=True, key="profile_chart_tab1")

    # Electrification legend badges
    badges = ""
    for sys in df["electrification"].unique():
        col   = elec_color(sys)
        km_e  = df[df["electrification"]==sys]["length_m"].sum()/1000
        icon  = "⚡" if sys!="NONE" else "🛢️"
        badges += (f'<span style="background:{col};color:white;padding:3px 10px;'
                   f'border-radius:12px;font-size:.8rem;font-weight:600;margin:2px;'
                   f'display:inline-block">{icon} {sys} · {km_e:.1f} km</span> ')
    st.markdown(badges, unsafe_allow_html=True)
    st.write("")

    # Route map
    map_df = df.dropna(subset=["lat","lon"])
    if not map_df.empty:
        with st.expander("🗺️ Route map", expanded=True):
            st.plotly_chart(make_route_map(df, st.session_state.combined_vias or [], parser),
                            use_container_width=True, key="route_map_tab1")

    # Speed summary table (pure Plotly — no matplotlib)
    with st.expander("⚡ Speed & electrification summary"):
        spd_df = df[["cum_km","station_name","stop_type","speed_kmh",
                      "gradient_perm","electrification","n_tracks"]].copy()
        spd_df = spd_df[spd_df["station_name"].str.strip().ne("")].reset_index(drop=True)
        def _sclr(v):
            lo,hi=30.0,200.0; t=max(0.0,min(1.0,(float(v)-lo)/(hi-lo)))
            r=int(220*(1-t)+34*t); g=int(38*(1-t)+197*t); b=int(38*(1-t)+94*t)
            return f"rgba({r},{g},{b},0.28)"
        icons = spd_df["stop_type"].map({"X":"🚉","R":"🛑"}).fillna("")
        labels = icons + " " + spd_df["station_name"].str.strip()

        fig_tbl = go.Figure(go.Table(
            columnwidth=[55,200,80,70,160,70],
            header=dict(values=["km","Waypoint","Speed [km/h]","Grad [‰]","Electrif.","Tracks"],
                        fill_color=C["dark"], font=dict(color="white",size=12),
                        align="left", height=28),
            cells=dict(values=[spd_df["cum_km"].round(2), labels,
                                spd_df["speed_kmh"].astype(int),
                                spd_df["gradient_perm"].round(1),
                                spd_df["electrification"], spd_df["n_tracks"]],
                        fill_color=[["#F8FAFC"]*len(spd_df), ["#F8FAFC"]*len(spd_df),
                                    [_sclr(v) for v in spd_df["speed_kmh"]],
                                    ["#F8FAFC"]*len(spd_df),
                                    ["#F8FAFC"]*len(spd_df),  # Safe solid color
                                    ["#F8FAFC"]*len(spd_df)],
                        font=dict(size=11), align="left", height=24)))
        fig_tbl.update_layout(height=min(60+len(spd_df)*26,440),
                               margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_tbl, use_container_width=True, key="speed_table_tab1")

    # Full data table
    with st.expander("📋 Full profile data table"):
        cols=["cum_km","station_name","stop_type","speed_kmh","gradient_perm",
              "electrification","recuperation","length_m","n_tracks"]
        cols=[c for c in cols if c in df.columns]
        ren=dict(cum_km="km [km]",station_name="Waypoint",stop_type="Stop",
                 speed_kmh="Speed [km/h]",gradient_perm="Grad [‰]",
                 electrification="Electrif.",recuperation="Recup.",
                 length_m="Length [m]",n_tracks="Tracks")
        st.dataframe(df[cols].rename(columns=ren), use_container_width=True,
                     hide_index=True, height=340)
        st.download_button("⬇️ Download profile CSV",
                            df[cols].to_csv(index=False).encode(),
                            "track_profile.csv","text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – PROFILE EDITOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_edit:
    st.markdown("### ✏️ Interactive Profile Editor")
    df = st.session_state.profile_df

    # Via waypoints
    st.markdown('<div class="sec">📍 Manual Routing Waypoints</div>', unsafe_allow_html=True)
    if st.session_state.get("scenario_mode") != "Single Journey":
        st.caption("ℹ️ **Note:** You are using a multi-leg scenario. Any manual routing waypoints added here will be applied to the **first leg** of the journey to force specific routing. Rebuilds profile automatically.")
    else:
        st.caption("Force the route through specific intermediate stations. Rebuilds profile automatically.")

    station_dict = {nm: oid for nm, oid in parser.station_list}
    station_names = list(station_dict.keys())

    cva, cvb = st.columns([3,1])
    via_sel_name = cva.selectbox("Search waypoint to add", options=station_names, index=None, placeholder="Type to search e.g. Pardubice...", label_visibility="collapsed", key="via_opt")

    if cvb.button("➕ Add", use_container_width=True, disabled=not via_sel_name):
        via_op = station_dict.get(via_sel_name)
        if via_op and via_op not in st.session_state.via_ops:
            st.session_state.via_ops.append(via_op)
            st.session_state.rebuild_profile = True
            st.rerun()

    if st.session_state.via_ops:
        st.markdown("**Manual routing waypoints (in order):**")
        for i, vid in enumerate(st.session_state.via_ops):
            info = parser.op_info.get(vid,{})
            e1,e2,e3,e4 = st.columns([0.25,0.25,3,1])
            e1.markdown(f"**{i+1}**")
            if e2.button("↑", key=f"up_{i}", disabled=i==0):
                st.session_state.via_ops[i], st.session_state.via_ops[i-1] = \
                    st.session_state.via_ops[i-1], st.session_state.via_ops[i]
                st.session_state.rebuild_profile = True
                st.rerun()
            e3.markdown(f"📍 **{info.get('name',vid)}** "
                         f"<span style='color:{C['grey']};font-size:.8rem'>"
                         f"{', '.join(info.get('types',[]))}</span>", unsafe_allow_html=True)
            if e4.button("✕ Remove", key=f"rm_{i}", use_container_width=True):
                st.session_state.via_ops.pop(i)
                st.session_state.rebuild_profile = True
                st.rerun()
        if st.button("🗑️ Clear all"):
            st.session_state.via_ops=[]
            st.session_state.rebuild_profile = True
            st.rerun()
    else:
        st.markdown('<div class="info-box">No manual waypoints. Router uses direct shortest path for each leg.</div>',
                    unsafe_allow_html=True)

    if df is not None:
        # Map
        st.markdown('<div class="sec">🗺️ Route Map</div>', unsafe_allow_html=True)
        map_df2 = df.dropna(subset=["lat","lon"])
        if not map_df2.empty:
            st.plotly_chart(make_route_map(df, st.session_state.combined_vias or [], parser),
                            use_container_width=True, key="route_map_tab2")

        # Override table
        st.markdown('<div class="sec">📝 Speed, Stop & Electrification Overrides</div>', unsafe_allow_html=True)
        st.caption("Edit speed limits, stop types, or electrification directly. Changes apply to the simulation only.")
        edit_cols = ["station_name","stop_type","speed_kmh","gradient_perm","electrification","length_m"]
        edit_cols = [c for c in edit_cols if c in df.columns]
        edited = st.data_editor(df[edit_cols].copy(), column_config={
            "station_name":    st.column_config.TextColumn("Waypoint",      disabled=True, width="large"),
            "stop_type":       st.column_config.SelectboxColumn("Stop",    options=["X","R",""],  width="small"),
            "speed_kmh":       st.column_config.NumberColumn("Speed [km/h]", min_value=0, max_value=350, width="small"),
            "gradient_perm":   st.column_config.NumberColumn("Grad [‰]",  disabled=True, width="small"),
            "electrification": st.column_config.SelectboxColumn("Electrif.", options=list(ELEC_COLORS.keys()), disabled=False, width="medium"),
            "length_m":        st.column_config.NumberColumn("Length [m]", disabled=True, width="small"),
        }, use_container_width=True, hide_index=True, num_rows="fixed", key="editor_tbl")

        if st.button("✅ Apply overrides", type="primary"):
            updated = df.copy()
            updated["speed_kmh"] = edited["speed_kmh"].values
            updated["stop_type"] = edited["stop_type"].values
            updated["electrification"] = edited["electrification"].values
            updated["recuperation"] = updated["electrification"].apply(lambda x: 0 if x == "NONE" else 1)

            # Recalculate the non-electrified gap ahead for the coasting physics engine
            if "ue_gap_m" in updated.columns:
                ue_gap_ahead = []
                for i in range(len(updated)):
                    if updated["electrification"].iloc[i] != "NONE":
                        ue_gap_ahead.append(0.0)
                        continue
                    gap, j = 0.0, i
                    while j < len(updated):
                        if updated["electrification"].iloc[j] != "NONE": break
                        gap += float(updated["length_m"].iloc[j])
                        j += 1
                    ue_gap_ahead.append(gap)
                updated["ue_gap_m"] = ue_gap_ahead

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

    # Fallback to handle cached sessions that lack the newer multi-leg dictionary keys
    legs_data   = rep.get("leg_results", [])
    total_stats = rep.get("total_stats", rep.get("stats", {}))
    total_net_w = rep.get("total_net_worst", total_stats.get("net_kwh", 0.0))

    _tr         = rep.get("traction", "ELECTRIC")
    _dd         = rep.get("diesel_d", 10.0)
    _eff        = rep.get("efficiency", 0.85)
    unit_lbl    = rep.get("unit_lbl", "kWh")

    consumed       = to_unit(total_stats, _tr, _dd, _eff)
    consumed_worst = to_unit({"net_kwh": total_net_w}, _tr, _dd, _eff)
    saved_amount   = consumed_worst - consumed

    df_p   = rep.get("sample_df", st.session_state.profile_df)
    tot_km = float(df_p["cum_km"].max()) if df_p is not None and not df_p.empty else 0.0

    j_time = total_stats.get("journey_time_s", 0)
    avg_spd = tot_km / (j_time / 3600) if j_time > 0 else 0
    sn_r   = df_p["station_name"].iloc[0]  if df_p is not None and not df_p.empty else ""
    en_r   = df_p["station_name"].iloc[-1] if df_p is not None and not df_p.empty else ""

    st.markdown(f"### ▶️ Simulation Results  ·  {rep.get('vehicle_name', 'Vehicle')}")

    st.markdown("#### Overall Scenario Statistics")
    kc2 = st.columns(5)
    kc2[0].markdown(kpi_card(fmt_dur(total_stats.get("journey_time_s", 0)), "Total Journey Time"), unsafe_allow_html=True)
    kc2[1].markdown(kpi_card(f"{consumed:.1f}", f"Total Consumed [{unit_lbl}]"), unsafe_allow_html=True)

    save_color = C["green"] if saved_amount > 0.01 else C["grey"]
    kc2[2].markdown(kpi_card(f"{saved_amount:.1f}", f"Total Saved [{unit_lbl}]", delta="vs. All Stops", color=save_color), unsafe_allow_html=True)

    kc2[3].markdown(kpi_card(f"{total_stats.get('regen_kwh', 0):.1f}", "Total Recuperated [kWh]"), unsafe_allow_html=True)

    # Safely compute total stops
    total_stops = sum(len(l.get("snames", [])) for l in legs_data)
    if not legs_data and "stop_names" in rep:
        total_stops = len(rep["stop_names"])

    kc2[4].markdown(kpi_card(str(total_stops), "Total Stops Served"), unsafe_allow_html=True)
    st.write("---")

    # Handle legacy single-leg runs visually gracefully
    if not legs_data and "hist" in rep:
        legs_data = [{
            "leg_num": 1,
            "start_name": sn_r,
            "end_name": en_r,
            "df_leg": df_p,
            "hist": rep.get("hist"),
            "stats": rep.get("stats", {}),
            "stats_worst": rep.get("stats_worst", rep.get("stats", {})),
            "snames": rep.get("stop_names", []),
            "total_possible_stops": len(rep.get("stop_names", []))
        }]

    for leg in legs_data:
        st.markdown(f"### Leg {leg.get('leg_num', 1)}: {leg.get('start_name', '')} ➔ {leg.get('end_name', '')}")

        leg_snames = leg.get("snames", [])
        total_possible = leg.get("total_possible_stops", len(leg_snames))

        st.markdown(f"**Stops served ({len(leg_snames)} / {total_possible}):** " + ("  →  ".join(f"*{s}*" for s in leg_snames) if leg_snames else "None"))

        leg_stats = leg.get("stats", {})
        leg_worst = leg.get("stats_worst", leg_stats)
        leg_cons  = to_unit(leg_stats, _tr, _dd, _eff)
        leg_cons_w= to_unit(leg_worst, _tr, _dd, _eff)
        leg_saved = leg_cons_w - leg_cons

        df_leg_ref = leg.get("df_leg", pd.DataFrame())
        leg_dist  = df_leg_ref["cum_km"].max() if not df_leg_ref.empty else 0.0
        j_time_leg = leg_stats.get("journey_time_s", 0)
        leg_avg_spd = leg_dist / (j_time_leg/3600) if j_time_leg > 0 else 0

        lk1, lk2, lk3, lk4, lk5 = st.columns(5)
        lk1.metric("Time", fmt_dur(j_time_leg))
        lk2.metric(f"Consumed [{unit_lbl}]", f"{leg_cons:.1f}")
        lk3.metric(f"Saved [{unit_lbl}]", f"{leg_saved:.1f}")
        lk4.metric("Avg Speed", f"{leg_avg_spd:.1f} km/h")

        if _tr == "DIESEL":
            lk5.metric(f"Specific Fuel", f"{leg_cons/leg_dist*1000:.1f} mL/km" if leg_dist > 0 else "0.0")
        else:
            lk5.metric(f"Specific Energy", f"{leg_stats.get('net_kwh', 0)/leg_dist:.3f} kWh/km" if leg_dist > 0 else "0.0")

        if leg.get("hist"):
            fig_s, fig_g, fig_e = make_kinematic_charts(leg["hist"], leg_snames, df_leg_ref, x_axis=x_axis_choice)
            st.plotly_chart(fig_s, use_container_width=True, key=f"kin_s_{leg.get('leg_num', 1)}")
            st.plotly_chart(fig_g, use_container_width=True, key=f"kin_g_{leg.get('leg_num', 1)}")
            st.plotly_chart(fig_e, use_container_width=True, key=f"kin_e_{leg.get('leg_num', 1)}")

        st.markdown("<hr>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 – MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mc_t:
    mc_data = st.session_state.mc_result
    df_p  = st.session_state.profile_df

    if mc_data is None:
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

    # Determine fallback state format for old cached sessions
    if isinstance(mc_data, pd.DataFrame):
        mc_overall = mc_data
        mc_legs = []
    else:
        mc_overall = mc_data.get("overall", pd.DataFrame())
        mc_legs = mc_data.get("legs", [])

    if mc_overall.empty:
        st.warning("No valid Monte Carlo data generated.")
        st.stop()

    ul = mc_overall["unit"].iloc[0]
    route_n = (f"{df_p['station_name'].iloc[0]} → {df_p['station_name'].iloc[-1]}"
               if df_p is not None else "")

    st.markdown(f"### 🎲 Monte Carlo Dashboard")

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
                f'<div class="info-box">ℹ️ Scenario route has <b>{nm} mandatory (X)</b> and '
                f'<b>{nr} request (R)</b> total stops. MC sweeps the probability of '
                f'serving request halts.</div>',
                unsafe_allow_html=True)

    st.caption(f"N = {int(mc_n)} simulation runs per probability level")

    def render_mc_dashboard(mc_df, title_str, route_desc, key_prefix):
        st.markdown(f"#### {title_str}")
        if route_desc:
            st.markdown(f"**Route:** {route_desc}")

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
                           f"mc_results_{key_prefix}.csv", "text/csv",
                           key=f"dl_mc_{key_prefix}")
        st.markdown("<br>", unsafe_allow_html=True)

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
            st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{key_prefix}")

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
            st.plotly_chart(fig_err, use_container_width=True, key=f"err_{key_prefix}")

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
        st.plotly_chart(fig_t, use_container_width=True, key=f"time_{key_prefix}")

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
            st.plotly_chart(fig_et, use_container_width=True, key=f"tradeoff_{key_prefix}")

        st.markdown("<hr>", unsafe_allow_html=True)


    # Output the Overall Simulation Output
    render_mc_dashboard(mc_overall, "Overall Scenario (Total Itinerary)", route_n, "overall")

    # Iterate over Legs
    if len(mc_legs) > 1:
        for leg in mc_legs:
            leg_route = f"{leg.get('start_name', 'Unknown')} → {leg.get('end_name', 'Unknown')}"
            render_mc_dashboard(leg["df"], f"Leg {leg.get('leg_num', '?')}", leg_route, f"leg_{leg.get('leg_num', '?')}")
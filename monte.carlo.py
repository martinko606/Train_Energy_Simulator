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
• Electric coasting through short incompatible/non-electrified gaps (configurable)
• Physics simulation with dynamic Davis equation & vehicle max-speed cap
• Kinematic chart with time/distance X-axis toggle & station annotations
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

    Speed strategy
    --------------
    Each meso-level segment can contain multiple track sub-elements.
    We take the MINIMUM speedSection value across all sub-elements per
    direction (most restrictive value any train must honour).
    For segments with no speedSection we fall back to
    linePerformance.maxSpeed, then 30 km/h (sidings/depots).
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

            # Safely assign normal and reverse gradients
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

            if length_m <= 0.0:
                length_m = 10.0  # Failsafe to prevent 0-cost shortcuts in Dijkstra

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

    # ── Dijkstra ──────────────────────────────────────────────────────────────
    def dijkstra(self, start: str, end: str,
                 penalty_m: float = 0.0, comp_sys: list[str] = None) -> tuple[float | None, list]:
        if start not in self.graph: return None, []
        tb  = itertools.count()
        pq  = [(0.0, next(tb), start, [])]
        vis: set = set()
        while pq:
            cost, _, node, path = heapq.heappop(pq)
            if node in vis: continue
            vis.add(node)
            if node == end: return cost, path
            for edge in self.graph.get(node, []):
                nxt = edge["to"]
                if nxt in vis: continue
                is_incomp = (comp_sys is not None) and (edge["electr"] not in comp_sys)
                pen = penalty_m if is_incomp else 0.0
                heapq.heappush(pq, (cost + edge["length_m"] + pen, next(tb), nxt,
                                    path + [dict(from_op=node, **edge)]))
        return None, []

    # ── Electrification analysis ───────────────────────────────────────────────
    def analyse_electrification(self, start_op: str, end_op: str, via_ops: list[str] = None, comp_sys: list[str] = None) -> dict:
        waypoints = [start_op] + (via_ops or []) + [end_op]

        p_normal = []
        for i in range(len(waypoints) - 1):
            _, path = self.dijkstra(waypoints[i], waypoints[i + 1], 0, comp_sys)
            if not path:
                return dict(normal_km=0, incomp_km=0, gateway_km=0,
                            penalised_km=0, penalised_incomp_km=0,
                            detour_km=0, saving_km=0, has_alternative=False)
            p_normal.extend(path)

        def _ue(path): return sum(self.seg_props.get(s["ne_id"],{}).get("length_m",0)
                                   for s in path if (comp_sys is not None and s["electr"] not in comp_sys)) / 1000
        def _tot(path): return sum(self.seg_props.get(s["ne_id"],{}).get("length_m",0)
                                    for s in path) / 1000
        normal_ue = _ue(p_normal); normal_tot = _tot(p_normal)
        gw_km = 0.0
        for s in p_normal:
            if comp_sys is not None and s["electr"] not in comp_sys: break
            gw_km += self.seg_props.get(s["ne_id"],{}).get("length_m",0)/1000

        p_pen = []
        for i in range(len(waypoints) - 1):
            _, path = self.dijkstra(waypoints[i], waypoints[i + 1], 100_000, comp_sys)
            if not path:
                p_pen = p_normal
                break
            p_pen.extend(path)

        pen_ue = _ue(p_pen); pen_tot = _tot(p_pen)
        saving = normal_ue - pen_ue; detour = pen_tot - normal_tot
        has_alt = saving >= 1.0 and detour <= max(saving * 10, 30)
        return dict(normal_km=round(normal_tot,1), incomp_km=round(normal_ue,1),
                    gateway_km=round(gw_km,1),
                    penalised_km=round(pen_tot,1), penalised_incomp_km=round(pen_ue,1),
                    detour_km=round(detour,1), saving_km=round(saving,1),
                    has_alternative=has_alt)

    # ── Profile builder ────────────────────────────────────────────────────────
    def build_profile(self, start_op: str, end_op: str,
                      via_ops: list[str] | None = None,
                      penalty_m: float = 0.0, comp_sys: list[str] = None) -> pd.DataFrame:
        waypoints = [start_op] + (via_ops or []) + [end_op]
        all_steps: list[dict] = []

        # Sequentially search and stitch the paths exactly through all given waypoints
        for i in range(len(waypoints) - 1):
            _, path = self.dijkstra(waypoints[i], waypoints[i + 1], penalty_m, comp_sys)
            if not path: return pd.DataFrame()
            all_steps.extend(path)

        if not all_steps: return pd.DataFrame()

        def _stop_type(oid: str) -> str:
            types = self.op_info.get(oid, {}).get("types", [])
            if any(t in MANDATORY_TYPES for t in types): return "X"
            if any(t in REQUEST_TYPES   for t in types): return "R"
            return ""

        rows: list[dict] = []; cum_m = 0.0
        for idx, step in enumerate(all_steps):
            from_op = step["from_op"]; seg = self.seg_props.get(step["ne_id"],{})
            fwd = step["forward"];  info = self.op_info.get(from_op,{})
            rows.append(dict(
                cum_km        = cum_m / 1000.0,
                op_id         = from_op,
                station_name  = info.get("name", from_op),
                stop_type     = _stop_type(from_op),
                lat           = info.get("lat"), lon=info.get("lon"),
                speed_kmh     = seg.get("speed_normal" if fwd else "speed_reverse", 30.0),
                gradient_perm = seg.get("grad_normal"  if fwd else "grad_reverse",  0.0),
                electrification=seg.get("electrification","NONE"),
                recuperation  = seg.get("recuperation",0),
                length_m      = seg.get("length_m",0.0),
                n_tracks      = seg.get("n_tracks","single"),
                braking_dist  = seg.get("braking_dist",0),
            ))
            cum_m += seg.get("length_m",0.0)

        last_fwd = all_steps[-1]["forward"]; last_seg = self.seg_props.get(all_steps[-1]["ne_id"],{})
        info_end = self.op_info.get(end_op,{})
        rows.append(dict(
            cum_km        = cum_m/1000.0, op_id=end_op,
            station_name  = info_end.get("name",end_op), stop_type="X",
            lat=info_end.get("lat"), lon=info_end.get("lon"),
            speed_kmh     = last_seg.get("speed_normal" if last_fwd else "speed_reverse",30.0),
            gradient_perm = last_seg.get("grad_normal"  if last_fwd else "grad_reverse", 0.0),
            electrification=last_seg.get("electrification","NONE"),
            recuperation  =last_seg.get("recuperation",0), length_m=0.0,
            n_tracks      =last_seg.get("n_tracks","single"),
            braking_dist  =last_seg.get("braking_dist",0),
        ))
        df = pd.DataFrame(rows); df["cum_km"] = df["cum_km"].round(4)
        return df

    def search_stations(self, query: str, max_results: int = 80) -> list[tuple[str, str]]:
        q = query.strip().lower()
        if not q: return []
        return [(oid, nm) for nm, oid in self.station_list if q in nm.lower()][:max_results]

    def op_name(self, op_id: str) -> str:
        return self.op_info.get(op_id,{}).get("name", op_id)


# ─────────────────────────────────────────────────────────────────────────────
#  TRACK PROFILE  (physics wrapper)
# ─────────────────────────────────────────────────────────────────────────────
class TrackProfile:
    def __init__(self, df: pd.DataFrame):
        self.df       = df.copy().reset_index(drop=True)
        self.total_km = float(df["cum_km"].max()) if not df.empty else 0.0
        self.segments = self._build_segs()
        self.stations = self._build_stations()

    def _build_segs(self) -> list[dict]:
        segs = []
        for i in range(len(self.df) - 1):
            r = self.df.iloc[i]
            segs.append(dict(
                km_start        = float(r["cum_km"]),
                km_end          = float(self.df.iloc[i+1]["cum_km"]),
                v_limit         = float(r["speed_kmh"]) / 3.6,
                grad            = float(r["gradient_perm"]) / 1000.0,
                electrification = str(r.get("electrification","NONE")),
                recuperation    = int(r.get("recuperation",0)),
            ))
        return segs

    def _build_stations(self) -> list[dict]:
        return [dict(name=str(r.get("station_name","")), km=float(r["cum_km"]),
                     type=str(r.get("stop_type","")).upper())
                for _, r in self.df.iterrows()
                if str(r.get("stop_type","")).upper() in ("X","R")
                and str(r.get("station_name","")).strip()]

    def seg_at(self, km: float) -> dict:
        for s in self.segments:
            if s["km_start"] <= km <= s["km_end"] + 1e-6: return s
        return self.segments[-1] if self.segments else {}

    def v_limit_span(self, front_km: float, rear_km: float) -> float:
        lo, hi = min(front_km, rear_km), max(front_km, rear_km)
        lims = [s["v_limit"] for s in self.segments
                if lo <= s["km_end"]+1e-6 and hi >= s["km_start"]-1e-6]
        return min(lims) if lims else (self.segments[-1]["v_limit"] if self.segments else 0.0)

    def get_incompatible_gap(self, current_km: float, comp_sys: list[str]) -> float:
        if comp_sys is None: return 0.0
        gap = 0.0
        curr_idx = -1
        for i, s in enumerate(self.segments):
            if s["km_start"] <= current_km <= s["km_end"] + 1e-6:
                curr_idx = i
                break
        if curr_idx == -1: return 0.0

        s = self.segments[curr_idx]
        if s["electrification"] in comp_sys: return 0.0

        gap += s["km_end"] - current_km
        for i in range(curr_idx + 1, len(self.segments)):
            s = self.segments[i]
            if s["electrification"] in comp_sys: break
            gap += s["km_end"] - s["km_start"]

        return gap * 1000.0

    @property
    def n_mandatory(self) -> int:
        return sum(1 for s in self.stations if s["type"] == "X")

    @property
    def n_request(self) -> int:
        return sum(1 for s in self.stations if s["type"] == "R")


# ─────────────────────────────────────────────────────────────────────────────
#  TRAIN SIMULATOR  (dynamic Davis equation + vehicle max-speed cap)
# ─────────────────────────────────────────────────────────────────────────────
class TrainSimulator:
    """
    1-D kinematic simulation.

    Davis resistance: F = A + B·v + C·v²  (mass-scaled coefficients)
      A = 20·m_t   (static/mechanical friction)
      B = 0.2·m_t  (flange/bearing speed-proportional)
      C = 2.5 + 0.03·L  (aerodynamic drag, length-dependent)
    where m_t = mass in tonnes, L = length in metres.

    Speed cap: track limit is further capped by the vehicle's own
    maximum permitted speed (max_speed_kmh).

    Electric coasting: gaps ≤ coast_threshold_m → coast on inertia.
    """

    def __init__(self, mass_kg, length_m, max_power_kw, aux_power_kw,
                 max_accel, max_decel, traction_type, efficiency,
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
        self.regen_eff  = 0.75 if traction_type == "ELECTRIC" else 0.0
        self.aux_w      = aux_power_kw  * 1_000.0
        self.max_w      = max_power_kw  * 1_000.0
        self.max_accel  = max_accel
        self.max_decel  = max_decel
        self.length_m   = length_m
        self.max_v      = max_speed_kmh / 3.6   # vehicle max speed [m/s]
        self.coast_threshold_m = coast_threshold_m
        self.comp_sys   = comp_sys

    def _res(self, v: float) -> float:
        return self.A + self.B * v + self.C * v * v

    def run(self, track: TrackProfile, stop_mode: str = "all",
            stop_prob: float = 1.0, dwell: float = 30.0,
            record: bool = False):
        total_m = track.total_km * 1000.0
        g = 9.8186

        # Build stops list (destination always included)
        stops_km: list[float] = []; stop_names: list[str] = []
        for st in track.stations:
            km = st["km"]
            if km <= 1e-3 or km >= track.total_km - 1e-3: continue
            will = (st["type"] == "X") or (
                st["type"] == "R" and (
                    stop_mode == "all" or
                    (stop_mode == "random" and random.random() <= stop_prob)))
            if will: stops_km.append(km); stop_names.append(st["name"])

        # Always stop at destination
        if track.total_km > 0:
            stops_km.append(track.total_km)
            stop_names.append("Destination")

        v = dist = e_j = r_j = t_s = 0.0
        dt = 1.0
        max_iters = int(total_m / 0.1) + 36_000
        iters = 0
        hist = {k: [] for k in ("time_s","km","v_kmh","v_limit_kmh",
                                 "gross_kwh","regen_kwh","net_kwh")} if record else None

        while dist < total_m:
            iters += 1
            if iters > max_iters:
                if coasting:
                    raise RuntimeError(f"Simulation stranded at km {km:.2f}. The train ran out of momentum while coasting through a neutral section. Increase entry speed, raise the coasting limit, or use a compatible train.")
                else:
                    raise RuntimeError(f"Simulation stalled at km {km:.2f}. The train lacks sufficient power to overcome the gradient and resistance.")

            km      = dist / 1000.0
            rear_km = max(0.0, km - self.length_m / 1000.0)
            seg     = track.seg_at(km)

            # --- BUG FIX: Prune passed stops to prevent desync/infinite stalls ---
            while stops_km and km > stops_km[0] + 0.1:
                stops_km.pop(0)

            # Track limit capped to vehicle max speed
            v_lim   = min(track.v_limit_span(km, rear_km), self.max_v)
            slope   = seg.get("grad", 0.0)
            electr  = seg.get("electrification", "NONE")
            recup   = seg.get("recuperation", 0) == 1

            # Electrification guard
            coasting = False
            if self.traction == "ELECTRIC" and (self.comp_sys is not None and electr not in self.comp_sys):
                gap_m = track.get_incompatible_gap(km, self.comp_sys)
                if gap_m <= self.coast_threshold_m:
                    coasting = True
                else:
                    raise RuntimeError(
                        f"⚡ Electric vehicle on incompatible track ({electr}) at km {km:.2f} "
                        f"(gap {gap_m/1000:.1f} km > coast limit {self.coast_threshold_m/1000:.1f} km). "
                        "Raise the coast threshold, enable 'Prefer compatible track', "
                        "or select a diesel vehicle.")

            f_grad    = self.mass_kg * g * slope
            eff_decel = max(0.05, self.max_decel + g * slope)

            # Braking look-ahead: use nearest upcoming stop
            max_safe = v_lim
            next_stop = stops_km[0] if stops_km else None
            if next_stop is not None:
                d2s = max(0.0, (next_stop - km) * 1000.0)
                max_safe = min(max_safe, math.sqrt(max(0.0, 2.0 * eff_decel * d2s)))

            if hist is not None:
                vl_front = min(track.v_limit_span(km, km), self.max_v)
                hist["time_s"].append(t_s); hist["km"].append(km)
                hist["v_kmh"].append(v * 3.6)
                hist["v_limit_kmh"].append(vl_front * 3.6)
                hist["gross_kwh"].append(e_j / 3_600_000.0)
                hist["regen_kwh"].append(r_j / 3_600_000.0)
                hist["net_kwh"].append((e_j - r_j) / 3_600_000.0)

            v_eff = max(v, 0.5); mech_p = reg_p = 0.0

            if coasting:
                f_res = self._res(v)
                v = max(0.0, v - (f_res + f_grad) / self.eff_mass)

            elif v > max_safe + 1e-4:
                f_res = self._res(v)
                nat_d = (f_res + f_grad) / self.eff_mass
                brk   = max(0.0, min(self.max_decel, (v - max_safe) - nat_d))
                if recup and not coasting: reg_p = self.eff_mass * brk * v * self.regen_eff
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

            e_j  += (mech_p / self.trac_eff + self.aux_w)
            r_j  += reg_p; dist += v * dt; t_s += dt

            # Dwell at stop
            if v < 0.5 and stops_km and abs(km - stops_km[0]) <= 0.1:
                v = 0.0
                if hist is not None:
                    for pass_no in range(2):
                        vl_front = min(track.v_limit_span(km,km), self.max_v)
                        hist["time_s"].append(t_s); hist["km"].append(km)
                        hist["v_kmh"].append(0.0)
                        hist["v_limit_kmh"].append(vl_front * 3.6)
                        hist["gross_kwh"].append(e_j / 3_600_000.0)
                        hist["regen_kwh"].append(r_j / 3_600_000.0)
                        hist["net_kwh"].append((e_j - r_j) / 3_600_000.0)
                        if pass_no == 0: e_j += self.aux_w * dwell; t_s += dwell
                else:
                    e_j += self.aux_w * dwell; t_s += dwell

                stops_km.pop(0)

                # Check destination reached
                if not stops_km or abs(km - track.total_km) <= 0.1:
                    dist = total_m; break

        return hist, stop_names, dict(
            gross_kwh=e_j/3_600_000.0, regen_kwh=r_j/3_600_000.0,
            net_kwh=(e_j-r_j)/3_600_000.0, journey_time_s=t_s)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fmt_dur(s: float) -> str:
    s = int(s); h, r = divmod(s,3600); m, sc = divmod(r,60)
    return f"{h:02d}h {m:02d}m {sc:02d}s" if h else f"{m:02d}m {sc:02d}s"

def to_unit(stats: dict, traction: str, density: float, eff: float) -> float:
    return stats["net_kwh"] / (density * eff) if traction == "DIESEL" else stats["net_kwh"]

def elec_color(label: str) -> str:
    for k, v in ELEC_COLORS.items():
        if k in label: return v
    return ELEC_COLORS["NONE"]

def kpi_card(val: str, lbl: str, delta: str = "", color: str = C["primary"]) -> str:
    d = (f'<div style="font-size:.72rem;color:{color};margin-top:2px;'
         f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{delta}</div>') if delta else ""
    return (f'<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;'
            f'padding:12px 14px;text-align:center;height:88px;display:flex;'
            f'flex-direction:column;justify-content:center;overflow:hidden">'
            f'<div style="font-size:1.35rem;font-weight:700;color:#1E3A5F;line-height:1.2">{val}</div>'
            f'<div style="font-size:.72rem;color:#64748B;margin-top:2px">{lbl}</div>{d}</div>')

def load_xml_from_upload(uploaded) -> str | None:
    # Use a deterministic temporary directory based on the app to prevent state wiping
    tmp_dir = os.path.join(tempfile.gettempdir(), "dypod_fixed_cache")
    os.makedirs(tmp_dir, exist_ok=True)

    ext = uploaded.name.lower().rsplit(".",1)[-1]
    if ext == "zip":
        zp = os.path.join(tmp_dir, uploaded.name)
        with open(zp,"wb") as f: f.write(uploaded.getbuffer())
        with zipfile.ZipFile(zp) as zf:
            xmls = [n for n in zf.namelist() if n.lower().endswith((".xml",".railml"))]
            if not xmls: return None
            extracted_path = os.path.join(tmp_dir, xmls[0])
            if not os.path.exists(extracted_path):
                zf.extract(xmls[0], tmp_dir)
            return extracted_path

    path = os.path.join(tmp_dir, uploaded.name)
    with open(path,"wb") as f: f.write(uploaded.getbuffer())
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  PROFILE CHART  (3-panel stepped: speed / gradient / electrification)
# ─────────────────────────────────────────────────────────────────────────────
def make_profile_chart(df: pd.DataFrame) -> go.Figure:
    """
    Three-panel chart with correctly stepped speed profile.
    Speed uses line_shape='hv' so each segment's limit is displayed as a
    horizontal plateau, with an instant vertical step at every km boundary.
    Station labels are thinned automatically to avoid overlap.
    Y-axis range is set to [0.75·min, 1.25·max] to make speed differences
    clearly visible even on slow regional lines.
    """
    stops_x = df[df["stop_type"] == "X"]
    stops_r = df[df["stop_type"] == "R"]
    total_km = float(df["cum_km"].max())

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Statutory Track Speed Limit [km/h]",
                        "Gradient [‰]  (↑ uphill  ↓ downhill)",
                        "Electrification"),
        row_heights=[0.48, 0.32, 0.20], vertical_spacing=0.055)

    # ── Speed stepped line ─────────────────────────────────────────────────
    hover_txt = []
    for _, row in df.iterrows():
        nm    = str(row.get("station_name","")).strip()
        stype = str(row.get("stop_type","")).strip()
        tag   = f"<b>{nm}</b><br>" if nm and nm != "nan" else ""
        stop_tag = {"X":" 🚉","R":" 🛑"}.get(stype,"")
        hover_txt.append(f"{tag}km {row['cum_km']:.2f}{stop_tag}<br>"
                          f"{row['speed_kmh']:.0f} km/h<br>"
                          f"Grad: {row['gradient_perm']:.1f} ‰<br>"
                          f"{row['electrification']}<extra></extra>")

    fig.add_trace(go.Scatter(x=df["cum_km"], y=df["speed_kmh"],
        mode="lines", name="Statutory Limit",
        line=dict(color=C["primary"], width=2.5, shape="hv"),
        fill="tozeroy", fillcolor=C["bg_blue"],
        hovertemplate=hover_txt), row=1, col=1)

    # Speed-change tick markers
    spd_changes = df[df["speed_kmh"].diff().abs() > 0.5].copy()
    if not spd_changes.empty:
        fig.add_trace(go.Scatter(x=spd_changes["cum_km"], y=spd_changes["speed_kmh"],
            mode="markers",
            marker=dict(symbol="line-ns", size=12, color=C["primary"],
                        line=dict(color=C["primary"], width=2)),
            showlegend=False,
            hovertemplate="km %{x:.2f}<br><b>%{y:.0f} km/h</b><extra></extra>"),
            row=1, col=1)

    # Station X markers — smart thinning (target ≤20 labelled)
    n_x = max(len(stops_x), 1)
    label_every = max(1, n_x // 20)
    labelled_x   = stops_x.iloc[::label_every]
    unlabelled_x = stops_x[~stops_x.index.isin(labelled_x.index)]

    if not labelled_x.empty:
        fig.add_trace(go.Scatter(x=labelled_x["cum_km"], y=labelled_x["speed_kmh"],
            mode="markers+text",
            marker=dict(symbol="diamond", size=9, color=C["accent"],
                        line=dict(color="white", width=1.5)),
            text=labelled_x["station_name"], textposition="top center",
            textfont=dict(size=7, color=C["dark"]), name="Station (X)",
            hovertemplate="<b>%{text}</b><br>km %{x:.2f}  %{y:.0f} km/h<extra></extra>"),
            row=1, col=1)

    if not unlabelled_x.empty:
        fig.add_trace(go.Scatter(x=unlabelled_x["cum_km"], y=unlabelled_x["speed_kmh"],
            mode="markers",
            marker=dict(symbol="diamond", size=6, color=C["accent"],
                        line=dict(color="white",width=1)),
            showlegend=False, text=unlabelled_x["station_name"],
            hovertemplate="<b>%{text}</b><br>km %{x:.2f}  %{y:.0f} km/h<extra></extra>"),
            row=1, col=1)

    if not stops_r.empty:
        fig.add_trace(go.Scatter(x=stops_r["cum_km"], y=stops_r["speed_kmh"],
            mode="markers", name="Halt (R)",
            marker=dict(symbol="circle", size=7, color=C["yellow"],
                        line=dict(color="white",width=1)),
            text=stops_r["station_name"],
            hovertemplate="<b>%{text}</b><br>km %{x:.2f}  %{y:.0f} km/h<extra></extra>"),
            row=1, col=1)

    # ── Gradient stepped fill ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=df["cum_km"], y=df["gradient_perm"].clip(lower=0),
        mode="lines", name="Uphill",
        line=dict(color=C["red"], width=0, shape="hv"),
        fill="tozeroy", fillcolor="rgba(220,38,38,0.55)",
        hovertemplate="km %{x:.2f}<br>+%{y:.1f} ‰<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["cum_km"], y=df["gradient_perm"].clip(upper=0),
        mode="lines", name="Downhill",
        line=dict(color=C["primary"], width=0, shape="hv"),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.55)",
        hovertemplate="km %{x:.2f}<br>%{y:.1f} ‰<extra></extra>"), row=2, col=1)

    # ── Electrification band — one bar per segment ─────────────────────────
    mid_km, seg_w, seg_c, seg_h = [], [], [], []
    for i in range(len(df)-1):
        x0, x1 = float(df["cum_km"].iloc[i]), float(df["cum_km"].iloc[i+1])
        mid_km.append((x0+x1)/2); seg_w.append(max(x1-x0,0.001))
        seg_c.append(elec_color(df["electrification"].iloc[i]))
        seg_h.append(df["electrification"].iloc[i])
    if len(df) > 0:
        mid_km.append(float(df["cum_km"].iloc[-1])); seg_w.append(0.001)
        seg_c.append(elec_color(df["electrification"].iloc[-1]))
        seg_h.append(df["electrification"].iloc[-1])

    fig.add_trace(go.Bar(x=mid_km, y=[1]*len(mid_km), width=seg_w,
        marker_color=seg_c, name="Electrification",
        hovertext=seg_h, hovertemplate="%{hovertext}<br>km %{x:.2f}<extra></extra>",
        showlegend=False), row=3, col=1)

    # Vertical lines at labelled stops
    for _, sr in labelled_x.iterrows():
        fig.add_vline(x=sr["cum_km"], line_width=0.7, line_dash="dot",
                      line_color="#CBD5E1", row=1, col=1)

    spd_min = float(df["speed_kmh"].min()); spd_max = float(df["speed_kmh"].max())
    fig.update_xaxes(title_text="Distance from departure [km]", row=3, col=1, gridcolor=C["light"])
    fig.update_yaxes(title_text="Speed [km/h]", row=1, col=1, gridcolor=C["light"],
                     range=[max(0, spd_min*0.75), spd_max*1.25])
    fig.update_yaxes(title_text="Gradient [‰]", row=2, col=1, gridcolor=C["light"],
                     zeroline=True, zerolinecolor="#CBD5E1")
    fig.update_yaxes(showticklabels=False, row=3, col=1, range=[0, 1.1])
    fig.update_layout(height=800, barmode="overlay", paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1),
        margin=dict(t=100, b=10, l=60, r=20),
        font=dict(family="Inter, sans-serif", size=12), hovermode="x unified")
    fig.update_xaxes(gridcolor=C["light"], showgrid=True)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  ROUTE MAP  (correct segment-by-segment colouring)
# ─────────────────────────────────────────────────────────────────────────────
def make_route_map(df: pd.DataFrame, via_ops: list,
                   parser: DYPODParser, show_halts: bool = True) -> go.Figure:
    """
    Route map with contiguous-run rendering so non-electrified sections
    (and all other electrification changes) are correctly coloured.
    """
    map_df = df.dropna(subset=["lat","lon"]).reset_index(drop=True)
    fig    = go.Figure()

    if not map_df.empty:
        runs: list[dict] = []
        cur_electr = None; cur_lats: list=[]; cur_lons: list=[]; cur_hover: list=[]

        for i in range(len(map_df)-1):
            ra = map_df.iloc[i]; rb = map_df.iloc[i+1]
            seg_e = str(ra["electrification"])
            if seg_e != cur_electr:
                if cur_lats:
                    runs.append(dict(electr=cur_electr, lats=cur_lats,
                                     lons=cur_lons, hover=cur_hover))
                cur_electr=seg_e; cur_lats=[float(ra["lat"])]; cur_lons=[float(ra["lon"])]
                cur_hover=[f"{seg_e}<br>km {ra['cum_km']:.2f}"]
            cur_lats.append(float(rb["lat"])); cur_lons.append(float(rb["lon"]))
            cur_hover.append(f"{seg_e}<br>km {rb['cum_km']:.2f}")

        if cur_lats: runs.append(dict(electr=cur_electr,lats=cur_lats,lons=cur_lons,hover=cur_hover))

        seen: set = set()
        for run in runs:
            show_leg = run["electr"] not in seen; seen.add(run["electr"])
            col = elec_color(run["electr"]); lw = 5 if run["electr"]!="NONE" else 4
            fig.add_trace(go.Scattermap(lat=run["lats"], lon=run["lons"], mode="lines",
                line=dict(width=lw, color=col), name=run["electr"], showlegend=show_leg,
                hovertext=run["hover"], hovertemplate="%{hovertext}<extra></extra>"))

    if show_halts:
        r_pts = map_df[map_df["stop_type"]=="R"]
        if not r_pts.empty:
            fig.add_trace(go.Scattermap(lat=r_pts["lat"].tolist(), lon=r_pts["lon"].tolist(),
                mode="markers", marker=dict(size=7, color=C["yellow"]), name="Halt (R)",
                customdata=r_pts["station_name"].values,
                hovertemplate="<b>%{customdata}</b><extra></extra>"))

    x_pts = map_df[map_df["stop_type"]=="X"]
    if not x_pts.empty:
        fig.add_trace(go.Scattermap(lat=x_pts["lat"].tolist(), lon=x_pts["lon"].tolist(),
            mode="markers+text", marker=dict(size=11, color=C["accent"]),
            text=x_pts["station_name"].tolist(), textposition="top right",
            textfont=dict(size=9, color=C["dark"]), name="Station (X)",
            customdata=x_pts[["cum_km","speed_kmh","gradient_perm"]].values,
            hovertemplate="<b>%{text}</b><br>km %{customdata[0]:.2f}<br>"
                           "%{customdata[1]:.0f} km/h  grad %{customdata[2]:.1f} ‰<extra></extra>"))

    for vid in (via_ops or []):
        info = parser.op_info.get(vid,{})
        if info.get("lat") and info.get("lon"):
            fig.add_trace(go.Scattermap(lat=[info["lat"]], lon=[info["lon"]],
                mode="markers+text", marker=dict(size=18,color=C["yellow"]),
                text=[info["name"]], textposition="top right",
                textfont=dict(size=10,color="#92400E"), name=f"Via: {info['name']}",
                showlegend=False))

    if not map_df.empty:
        for row_lat,row_lon,row_nm,tpos,col,leg in [
            (float(map_df["lat"].iloc[0]),  float(map_df["lon"].iloc[0]),
             map_df["station_name"].iloc[0],  "top right", C["green"],"Departure"),
            (float(map_df["lat"].iloc[-1]), float(map_df["lon"].iloc[-1]),
             map_df["station_name"].iloc[-1], "top left",  C["red"],  "Arrival"),
        ]:
            fig.add_trace(go.Scattermap(lat=[row_lat], lon=[row_lon],
                mode="markers+text", marker=dict(size=20,color=col),
                text=[row_nm], textposition=tpos,
                textfont=dict(size=11,color=C["dark"],family="Inter, sans-serif"),
                name=leg, hovertemplate="<b>%{text}</b><extra></extra>"))

    lat_c = float(map_df["lat"].mean()) if not map_df.empty else 50.0
    lon_c = float(map_df["lon"].mean()) if not map_df.empty else 15.5
    fig.update_layout(map=dict(style="open-street-map",
                               center=dict(lat=lat_c,lon=lon_c),zoom=7),
        height=540, margin=dict(l=0,r=0,t=0,b=0),
        legend=dict(bgcolor="rgba(255,255,255,0.92)",bordercolor="#E2E8F0",
                    borderwidth=1,x=0.01,y=0.99,font=dict(size=11)))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  KINEMATIC CHART  (speed + energy with time/distance X-axis toggle)
# ─────────────────────────────────────────────────────────────────────────────
def make_kinematic_chart(hist: dict, stop_names: list[str],
                         df_profile: pd.DataFrame,
                         x_axis: str = "Distance (km)") -> go.Figure:
    """
    Two-panel kinematic plot:
      Top   — actual speed vs speed limit (stepped)
      Bottom — gross / recuperated / net energy
    X-axis toggles between cumulative distance [km] and elapsed time [MM:SS].
    Station positions are annotated with rotated labels.
    """
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
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.18,
                        subplot_titles=("Simulated Train Speed [km/h]",
                                        "Cumulative Energy [kWh]"))

    # Speed traces
    fig.add_trace(go.Scatter(x=x_data, y=hist["v_limit_kmh"], name="Speed Limit",
        line=dict(color=C["red"], dash="dash", width=1.8, shape="hv")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_data, y=hist["v_kmh"], name="Actual Speed",
        line=dict(color=C["primary"], width=2.5),
        fill="tozeroy", fillcolor=C["bg_blue"]), row=1, col=1)

    # Energy traces
    fig.add_trace(go.Scatter(x=x_data, y=hist["gross_kwh"], name="Gross Energy",
        line=dict(color=C["yellow"], width=2),
        fill="tozeroy", fillcolor="rgba(202,138,4,0.10)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_data, y=hist["regen_kwh"], name="Recuperated",
        line=dict(color=C["green"], width=2, dash="dash"),
        fill="tozeroy", fillcolor="rgba(22,163,74,0.10)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_data, y=hist["net_kwh"], name="Net Energy",
        line=dict(color=C["secondary"], width=2.5)), row=2, col=1)

    # Station annotations
    v_max = max(max(hist["v_limit_kmh"]), max(hist["v_kmh"])) * 1.05 if hist["v_kmh"] else 10
    e_max = max(hist["gross_kwh"]) * 1.05 if hist["gross_kwh"] else 1

    km_arr = np.array(hist["km"])
    all_stops_df = df_profile[df_profile["stop_type"].isin(["X","R"])] if df_profile is not None else pd.DataFrame()

    shapes: list[dict] = []; annotations: list[dict] = []
    for _, srow in all_stops_df.iterrows():
        skm   = float(srow["cum_km"])
        stype = srow["stop_type"]
        sname = srow["station_name"]
        if not (km_arr.min()-0.1 <= skm <= km_arr.max()+0.1): continue
        color = C["grey"] if stype == "X" else (C["primary"] if sname in stops_set else C["red"])
        idx   = int(np.argmin(np.abs(km_arr - skm)))
        x_pos = time_dt[idx] if use_time else float(dist_km[idx])

        shapes += [
            dict(type="line", xref="x", yref="y",  x0=x_pos, x1=x_pos,
                 y0=0, y1=v_max, line=dict(color=color, width=1, dash="dot"), layer="below"),
            dict(type="line", xref="x", yref="y2", x0=x_pos, x1=x_pos,
                 y0=0, y1=e_max, line=dict(color=color, width=1, dash="dot"), layer="below"),
        ]
        for yref in ("y domain", "y2 domain"):
            annotations.append(dict(
                x=x_pos, y=-0.18, xref="x", yref=yref,
                text=sname, showarrow=False, font=dict(size=10, color=color),
                xanchor="right", yanchor="top", textangle=-45))

    fig.update_layout(
        height=820, margin=dict(l=60, r=40, t=70, b=140), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0,
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1),
        shapes=shapes, annotations=annotations,
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12),
    )
    fig.update_xaxes(title_text=x_title, range=[x_min, x_max],
                     tickformat=x_fmt, showgrid=True, gridcolor=C["light"])
    fig.update_yaxes(title_text="Speed [km/h]",  row=1, col=1, showgrid=True, gridcolor=C["light"])
    fig.update_yaxes(title_text="Energy [kWh]",  row=2, col=1, showgrid=True, gridcolor=C["light"])
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT APP
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DYPOD Simulator", page_icon="🚆", layout="wide")
st.markdown("""<style>
[data-testid="stSidebar"]{background:#F1F5F9}
.stButton>button{border-radius:8px;font-weight:600;transition:all .15s}
.stButton>button:hover{transform:translateY(-1px);box-shadow:0 2px 8px rgba(0,0,0,.15)}
.block-container{padding-top:3.5rem; padding-bottom:2rem}
.stTabs [data-baseweb="tab"]{font-weight:600;font-size:.92rem}
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

if "rebuild_profile" not in st.session_state:
    st.session_state.rebuild_profile = False
for _k in ("parser","xml_path","profile_df","via_ops","rep_result","mc_result",
           "elec_analysis","op_start","op_end","comp_sys"):
    if _k not in st.session_state: st.session_state[_k] = None
if not isinstance(st.session_state.via_ops, list): st.session_state.via_ops = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚆 DYPOD Simulator")
    st.caption("Czech railway track profile & energy analysis")
    st.markdown("---")

    # 1. File
    st.markdown('<div class="sec">📂 Load railML</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload railML or zip", type=["xml","railml","zip"],
                                 label_visibility="collapsed")
    xml_path = None
    if uploaded:
        # Prevent constant state-wiping by checking file name and size
        file_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("last_uploaded_id") != file_id:
            st.session_state.last_uploaded_id = file_id
            xp = load_xml_from_upload(uploaded)
            for k in ("parser","xml_path","profile_df","via_ops","rep_result","mc_result","elec_analysis","comp_sys"):
                st.session_state[k] = None if k != "via_ops" else []
            st.session_state.xml_path = xp
        xml_path = st.session_state.xml_path
    else:
        local = sorted(glob.glob("*.xml")+glob.glob("*.railml")+glob.glob("*.zip")+
                       glob.glob("/tmp/*.xml")+glob.glob("/tmp/*.railml"))
        if local:
            sel = st.selectbox("Or pick local file", local, label_visibility="collapsed")
            if sel != st.session_state.xml_path:
                for k in ("parser","xml_path","profile_df","via_ops","rep_result","mc_result","elec_analysis","comp_sys"):
                    st.session_state[k] = None if k != "via_ops" else []
                st.session_state.xml_path = sel
                st.session_state.last_uploaded_id = None
            xml_path = sel

    @st.cache_resource(show_spinner="Parsing railML infrastructure (~5 s)…")
    def _load(path: str) -> DYPODParser:
        if path.lower().endswith(".zip"):
            tmp = tempfile.mkdtemp()
            with zipfile.ZipFile(path) as zf:
                xmls=[n for n in zf.namelist() if n.lower().endswith((".xml",".railml"))]
                zf.extract(xmls[0], tmp); path=os.path.join(tmp,xmls[0])
        return DYPODParser(path)

    parser: DYPODParser | None = None
    if xml_path:
        try:
            parser = _load(xml_path)
            st.success(f"✅ {parser.parse_time} s · **{len(parser.op_info):,}** OPs · "
                        f"**{len(parser.seg_props):,}** segments · "
                        f"**{len(parser.station_list):,}** passenger stops")
        except Exception as exc:
            st.error(f"Parse error: {exc}"); st.stop()
    else:
        st.info("Upload a DYPOD railML file to begin."); st.stop()

    # 2. Vehicle
    st.markdown('<div class="sec">🚃 Vehicle</div>', unsafe_allow_html=True)
    veh = st.selectbox("Preset", ["Custom"] + list(PREDEFINED_VEHICLES.keys()),
                       label_visibility="collapsed")
    if veh == "Custom":
        traction   = st.selectbox("Traction", ["DIESEL","ELECTRIC"])
        cv1, cv2   = st.columns(2)
        mass       = cv1.number_input("Mass (kg)",    value=100_000, step=5_000)
        length     = cv2.number_input("Length (m)",   value=40,      step=5)
        power      = cv1.number_input("Power (kW)",   value=500,     step=50)
        aux_power  = cv2.number_input("Aux (kW)",     value=40,      step=5)
        max_speed  = st.number_input("Max speed (km/h)", value=160, min_value=30, max_value=350)
        accel      = st.slider("Accel (m/s²)", 0.2, 1.5, 0.6, 0.05)
        decel      = st.slider("Brake (m/s²)", 0.4, 1.5, 0.9, 0.05)
        efficiency = st.slider("Efficiency (%)", 15, 95, 35) / 100.0
        diesel_d   = st.number_input("Diesel density (kWh/L)", value=10.0) if traction=="DIESEL" else 10.0
    else:
        p = PREDEFINED_VEHICLES[veh]
        traction, mass, length = p["traction"], p["mass"], p["length"]
        power, aux_power       = p["power"], p["aux_power"]
        accel, decel, efficiency, max_speed = p["accel"], p["decel"], p["efficiency"], p["max_speed"]
        diesel_d = st.number_input("Diesel density (kWh/L)", value=10.0) if traction=="DIESEL" else 10.0
        st.info(f"**{veh}** |  {traction}  |  {max_speed} km/h  |  "
                f"{power} kW  |  {mass:,} kg")
    unit_lbl = "L fuel" if traction == "DIESEL" else "kWh"

    if traction == "ELECTRIC":
        if veh == "Custom":
            comp_sys = st.multiselect("Compatible catenary systems", ["3,000V/0Hz", "25,000V/50Hz", "1,500V/0Hz", "15,000V/16.7Hz"], default=["3,000V/0Hz", "25,000V/50Hz"])
        else:
            comp_sys = st.multiselect("Compatible catenary systems", ["3,000V/0Hz", "25,000V/50Hz", "1,500V/0Hz", "15,000V/16.7Hz"], default=p.get("systems", []))
    else:
        comp_sys = None

    # 3. Route
    st.markdown('<div class="sec">🗺️ Route</div>', unsafe_allow_html=True)

    station_dict = {nm: oid for nm, oid in parser.station_list}
    station_names = list(station_dict.keys())

    if station_names:
        op_start_name = st.selectbox("🔍 Departure station", station_names, index=0)
        op_end_name   = st.selectbox("🔍 Arrival station", station_names, index=len(station_names)-1)

        op_start = station_dict.get(op_start_name)
        op_end   = station_dict.get(op_end_name)
    else:
        op_start, op_end = None, None
        st.warning("No passenger stations found in the data.")

    if traction == "ELECTRIC":
        st.markdown("**⚡ Compatible route preference**")
        elec_reroute = st.toggle("Prefer compatible track", value=False,
                                  help="Penalises incompatible/non-electrified segments in route search.")
        pen_km = st.slider("Detour tolerance (km per incompatible segment)", 10, 500, 100, 10) if elec_reroute else 0
    else:
        elec_reroute = False
        pen_km = 0

    btn_profile = st.button("🗺️  Build Track Profile", use_container_width=True, type="primary",
                             disabled=not(op_start and op_end and op_start != op_end))


    # 4. Simulation
    st.markdown('<div class="sec">⚙️ Simulation</div>', unsafe_allow_html=True)
    dwell     = st.number_input("Station dwell (s)", value=30, step=5)
    stop_mode = st.radio("Stop mode", ["all","random","none"],
                          format_func={"all":"All stops","random":"Stochastic","none":"Express"}.get,
                          horizontal=True)
    stop_prob = st.slider("Request-stop probability", 0.0, 1.0, 0.5, 0.05,
                           disabled=(stop_mode != "random"))

    coast_threshold_m = 500.0
    if traction == "ELECTRIC":
        st.markdown("**⚡ Electric coasting**")
        coast_km = st.slider("Max coastable incompatible gap (km)", 0.1, 10.0, 0.5, 0.1,
                              help="Electric vehicles coast (no traction, aux only) through "
                                   "incompatible/non-electrified gaps shorter than this.")
        coast_threshold_m = coast_km * 1000.0

    # 5. Monte Carlo
    st.markdown('<div class="sec">🎲 Monte Carlo</div>', unsafe_allow_html=True)
    mc_n     = st.number_input("Runs per probability", 20, 500, 100, 10)
    mc_probs = st.multiselect("Probabilities to sweep",
                               [1.0,0.8,0.6,0.4,0.2,0.0],
                               default=[1.0,0.8,0.6,0.4,0.2,0.0])

    st.markdown('<div class="sec">📊 Display Options</div>', unsafe_allow_html=True)
    x_axis_choice = st.radio("Kinematic X-axis", ["Distance (km)", "Time (MM:SS)"])

    cb1, cb2 = st.columns(2)
    btn_run = cb1.button("▶️ Run",  use_container_width=True,
                          disabled=st.session_state.profile_df is None)
    btn_mc  = cb2.button("🎲 MC",   use_container_width=True,
                          disabled=st.session_state.profile_df is None)


# ── Actions ───────────────────────────────────────────────────────────────────
if (btn_profile or st.session_state.rebuild_profile) and op_start and op_end:
    st.session_state.rebuild_profile = False
    pen = pen_km * 1000.0 if elec_reroute else 0.0
    with st.spinner("Finding path and building profile…"):
        ea  = parser.analyse_electrification(op_start, op_end, via_ops=st.session_state.via_ops or [], comp_sys=comp_sys)
        df  = parser.build_profile(op_start, op_end,
                                    via_ops=st.session_state.via_ops or [],
                                    penalty_m=pen, comp_sys=comp_sys)
    if df is None or df.empty: st.error("No path found between the selected stations.")
    else:
        st.session_state.update(profile_df=df, op_start=op_start, op_end=op_end,
                                 elec_analysis=ea, rep_result=None, mc_result=None,
                                 comp_sys=comp_sys)

if btn_run and st.session_state.profile_df is not None:
    with st.spinner("Running kinematic physics simulation…"):
        try:
            track = TrackProfile(st.session_state.profile_df)
            sim   = TrainSimulator(mass, length, power, aux_power, accel, decel,
                                   traction, efficiency, max_speed, coast_threshold_m,
                                   comp_sys=st.session_state.comp_sys)
            hist, snames, stats = sim.run(track, stop_mode=stop_mode, stop_prob=stop_prob,
                                           dwell=dwell, record=True)
            st.session_state.rep_result = dict(hist=hist, stop_names=snames, stats=stats,
                traction=traction, diesel_d=diesel_d, efficiency=efficiency,
                unit_lbl=unit_lbl, vehicle_name=veh, dwell=dwell,
                coast_threshold_m=coast_threshold_m)
        except RuntimeError as e:
            st.error(str(e))

if btn_mc and st.session_state.profile_df is not None and mc_probs:
    try:
        track = TrackProfile(st.session_state.profile_df)
        sim   = TrainSimulator(mass, length, power, aux_power, accel, decel,
                               traction, efficiency, max_speed, coast_threshold_m,
                               comp_sys=st.session_state.comp_sys)
        rows_mc = []
        total_runs = len(mc_probs) * int(mc_n)
        pb = st.sidebar.progress(0.0, text="Monte Carlo running…")
        done = 0
        for p_val in sorted(mc_probs, reverse=True):
            sm = "all" if p_val==1.0 else "none" if p_val==0.0 else "random"
            e_list, t_list = [], []
            for _ in range(int(mc_n)):
                _, _, st_ = sim.run(track, sm, p_val, dwell)
                e_list.append(to_unit(st_, traction, diesel_d, efficiency))
                t_list.append(st_["journey_time_s"])
                done += 1
                pb.progress(done / total_runs,
                             text=f"MC running… {done}/{total_runs}")
            rows_mc.append(dict(prob=f"{int(p_val*100)}%", p_num=p_val,
                mean_e=np.mean(e_list),  std_e=np.std(e_list),
                min_e=np.min(e_list),    max_e=np.max(e_list),
                mean_t=np.mean(t_list),  min_t=np.min(t_list), max_t=np.max(t_list)))
        pb.empty()
        mc_df = pd.DataFrame(rows_mc)
        mc_df["savings"] = (mc_df["mean_e"].max() - mc_df["mean_e"]).clip(lower=0)
        mc_df["unit"]    = unit_lbl
        st.session_state.mc_result = mc_df
    except RuntimeError as e:
        if 'pb' in dir(): pb.empty()
        st.error(str(e))


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_prof, tab_edit, tab_run_t, tab_mc_t = st.tabs([
    "🗺️  Track Profile", "✏️  Profile Editor",
    "▶️  Kinematic Simulation", "🎲  Monte Carlo",
])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – TRACK PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_prof:
    df  = st.session_state.profile_df
    ea  = st.session_state.elec_analysis
    cs  = st.session_state.comp_sys

    if df is None:
        st.markdown("### 👈 Search for stations in the sidebar and click **Build Track Profile**")
        st.info("Stitches meso-level railML segments into a direction-aware profile: "
                "stepped speed limits, gradient, electrification, GPS coordinates.")
        st.stop()

    total_km = float(df["cum_km"].max())
    stops_x  = df[df["stop_type"]=="X"]; stops_r = df[df["stop_type"]=="R"]

    incomp_km = sum(s.get("length_m", 0) for s in df.to_dict('records') if cs is not None and s.get("electrification") not in cs) / 1000
    comp_km   = total_km - incomp_km
    sn = df["station_name"].iloc[0]; en = df["station_name"].iloc[-1]

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
    kc[0].markdown(kpi_card(f"{total_km:.1f} km","Route length"), unsafe_allow_html=True)
    kc[1].markdown(kpi_card(str(len(stops_x)),"Mandatory stops (X)"), unsafe_allow_html=True)
    kc[2].markdown(kpi_card(str(len(stops_r)),"Request halts (R)"), unsafe_allow_html=True)
    kc[3].markdown(kpi_card(f"{df['speed_kmh'].max():.0f} km/h","Max speed"), unsafe_allow_html=True)
    kc[4].markdown(kpi_card(f"{df['gradient_perm'].abs().max():.0f} ‰","Max gradient"), unsafe_allow_html=True)

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
    st.markdown(badges, unsafe_allow_html=True); st.write("")

    # Route map
    map_df = df.dropna(subset=["lat","lon"])
    if not map_df.empty:
        with st.expander("🗺️ Route map", expanded=True):
            st.plotly_chart(make_route_map(df, st.session_state.via_ops or [], parser),
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
    st.markdown('<div class="sec">📍 Via Waypoints</div>', unsafe_allow_html=True)
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
        st.markdown("**Via waypoints (in order):**")
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
        st.markdown('<div class="info-box">No via-waypoints. Router uses direct shortest path.</div>',
                    unsafe_allow_html=True)

    if df is not None:
        # Map
        st.markdown('<div class="sec">🗺️ Route Map</div>', unsafe_allow_html=True)
        map_df2 = df.dropna(subset=["lat","lon"])
        if not map_df2.empty:
            st.plotly_chart(make_route_map(df, st.session_state.via_ops or [], parser),
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
#  TAB 3 – KINEMATIC SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_run_t:
    rep = st.session_state.rep_result
    if rep is None:
        st.info("👈 Build a profile, then click **▶️ Run** in the sidebar.")
        st.stop()

    hist   = rep["hist"]; stats=rep["stats"]; snames=rep["stop_names"]
    _tr    = rep["traction"]; _dd=rep["diesel_d"]; _eff=rep["efficiency"]
    consumed = to_unit(stats, _tr, _dd, _eff)
    df_p   = st.session_state.profile_df
    tot_km = float(df_p["cum_km"].max()) if df_p is not None else 0
    avg_spd = tot_km/(stats["journey_time_s"]/3600) if stats["journey_time_s"]>0 else 0
    sn_r   = df_p["station_name"].iloc[0]  if df_p is not None else ""
    en_r   = df_p["station_name"].iloc[-1] if df_p is not None else ""

    st.markdown(f"### ▶️ {sn_r}  →  {en_r}  ·  {rep['vehicle_name']}")

    kc2 = st.columns(6)
    kc2[0].markdown(kpi_card(fmt_dur(stats["journey_time_s"]),"Journey time"), unsafe_allow_html=True)
    kc2[1].markdown(kpi_card(f"{consumed:.1f}",f"Net [{unit_lbl}]"), unsafe_allow_html=True)
    kc2[2].markdown(kpi_card(f"{stats['gross_kwh']:.1f}","Gross [kWh]"), unsafe_allow_html=True)
    kc2[3].markdown(kpi_card(f"{stats['regen_kwh']:.1f}","Recuperated [kWh]"), unsafe_allow_html=True)
    kc2[4].markdown(kpi_card(f"{avg_spd:.1f} km/h","Avg journey speed"), unsafe_allow_html=True)
    kc2[5].markdown(kpi_card(str(len(snames)),"Stops served"), unsafe_allow_html=True)
    st.write("")

    if snames:
        st.markdown("**Stops served:** " + "  →  ".join(f"*{s}*" for s in snames if s != "Destination"))
    else:
        st.caption("No intermediate stops served (express run).")

    if hist:
        fig_kin = make_kinematic_chart(hist, snames, df_p, x_axis=x_axis_choice)
        st.plotly_chart(fig_kin, use_container_width=True, key="kin_chart_tab3")

        csv_kin = pd.DataFrame(hist).to_csv(index=False).encode()
        st.download_button("⬇️ Download telemetry CSV", csv_kin,
                            "kinematic_telemetry.csv","text/csv")

        if stats["gross_kwh"] > 0 and tot_km > 0:
            rp   = stats["regen_kwh"]/stats["gross_kwh"]*100
            spec = stats["net_kwh"]/tot_km
            st.markdown("---")
            ec1,ec2,ec3,ec4 = st.columns(4)
            ec1.metric("Recuperation share",  f"{rp:.1f}%")
            ec2.metric("Specific net energy", f"{spec:.3f} kWh/km")
            if _tr == "DIESEL":
                ec3.metric("Specific fuel", f"{consumed/tot_km:.3f} L/km")
                ec4.metric("CO₂ (est.)", f"{consumed*2.65:.0f} kg",
                            help="Diesel combustion ≈ 2.65 kg CO₂/L")
            else:
                ec3.metric("Net kWh/km", f"{spec:.3f}")
                ec4.metric("Regen saved", f"{stats['regen_kwh']:.1f} kWh")

        # Speed + energy distributions
        with st.expander("📊 Distributions"):
            dd1, dd2 = st.columns(2)
            with dd1:
                fig_sv = go.Figure(go.Histogram(x=hist["v_kmh"], nbinsx=40,
                    marker_color=C["primary"], opacity=0.8))
                fig_sv.update_layout(title="Time at speed", xaxis_title="Speed [km/h]",
                    yaxis_title="Seconds", height=280, paper_bgcolor="white",
                    plot_bgcolor="white", margin=dict(t=40,b=40,l=50,r=10))
                st.plotly_chart(fig_sv, use_container_width=True, key="hist_tab3")
            with dd2:
                wf_v = [stats["gross_kwh"],-stats["regen_kwh"],stats["net_kwh"]]
                fig_wf = go.Figure(go.Waterfall(orientation="v",
                    measure=["absolute","relative","total"],
                    x=["Gross","Recuperated","Net"], y=wf_v,
                    connector=dict(line=dict(color="#CBD5E1")),
                    decreasing=dict(marker=dict(color=C["green"])),
                    increasing=dict(marker=dict(color=C["red"])),
                    totals=dict(marker=dict(color=C["primary"])),
                    text=[f"{v:.1f} kWh" for v in wf_v], textposition="outside"))
                fig_wf.update_layout(title="Energy waterfall [kWh]", height=280,
                    paper_bgcolor="white", plot_bgcolor="white",
                    margin=dict(t=40,b=40,l=50,r=10))
                st.plotly_chart(fig_wf, use_container_width=True, key="waterfall_tab3")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 – MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════════
with tab_mc_t:
    mc_df = st.session_state.mc_result
    if mc_df is None:
        st.info("👈 Build a profile, then click **🎲 MC** in the sidebar.")
        df_p = st.session_state.profile_df
        if df_p is not None:
            tr = TrackProfile(df_p)
            if tr.n_request == 0:
                st.markdown('<div class="warn-box">⚠️ <b>No request stops (R)</b> on this route — '
                            'Monte Carlo variance will be zero. All stops are mandatory stations (X). '
                            'Try a route with <i>stoppingPoint</i>-type halts for stochastic results.</div>',
                            unsafe_allow_html=True)
        st.stop()

    ul    = mc_df["unit"].iloc[0]
    df_p  = st.session_state.profile_df
    route_n = (f"{df_p['station_name'].iloc[0]} → {df_p['station_name'].iloc[-1]}"
               if df_p is not None else "")

    st.markdown(f"### 🎲 Monte Carlo — {route_n}")

    if df_p is not None:
        tr = TrackProfile(df_p)
        nm, nr = tr.n_mandatory, tr.n_request
        if nr == 0:
            st.markdown('<div class="warn-box">⚠️ All stops are mandatory (X) — std = 0.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="info-box">ℹ️ <b>{nm} mandatory (X)</b> and '
                         f'<b>{nr} request (R)</b> stops. MC sweeps request-stop probability.</div>',
                         unsafe_allow_html=True)
    st.caption(f"N = {int(mc_n)} runs per probability level")

    disp = mc_df.copy()
    disp["Mean time"] = disp["mean_t"].apply(fmt_dur)
    disp["Fastest"]   = disp["min_t"].apply(fmt_dur)
    disp["Slowest"]   = disp["max_t"].apply(fmt_dur)
    show_map = {"prob":"Probability","mean_e":f"Mean [{ul}]","std_e":f"Std [{ul}]",
                "min_e":f"Min [{ul}]","max_e":f"Max [{ul}]",
                "savings":f"Savings [{ul}]","Mean time":"Mean time",
                "Fastest":"Fastest","Slowest":"Slowest"}
    d2 = disp.rename(columns=show_map)
    st.dataframe(d2[[v for v in show_map.values() if v in d2.columns]],
                 use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download CSV", mc_df.to_csv(index=False).encode(),
                        "mc_results.csv","text/csv")
    st.markdown("---")

    mc1, mc2 = st.columns(2)
    with mc1:
        fig_bar = go.Figure(go.Bar(x=mc_df["prob"], y=mc_df["savings"],
            marker=dict(color=mc_df["savings"], colorscale="Blues",
                        showscale=True, colorbar=dict(title=ul, len=0.8)),
            text=mc_df["savings"].round(2), texttemplate="%{text}", textposition="outside",
            hovertemplate="<b>%{x}</b><br>Savings: %{y:.2f} "+ul+"<extra></extra>"))
        fig_bar.update_layout(title=f"<b>Energy savings vs. all-stops</b> [{ul}]",
            xaxis_title="Request-stop probability", yaxis_title=f"Savings [{ul}]",
            height=400, paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(t=60,b=50,l=60,r=20),
            font=dict(family="Inter, sans-serif",size=12), yaxis=dict(gridcolor=C["light"]))
        st.plotly_chart(fig_bar, use_container_width=True, key="bar_tab4")

    with mc2:
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(
            x=list(mc_df["prob"])+list(mc_df["prob"].iloc[::-1]),
            y=list(mc_df["max_e"])+list(mc_df["min_e"].iloc[::-1]),
            fill="toself", fillcolor=C["bg_blue"],
            line=dict(color="rgba(255,255,255,0)"), name="Min–Max range"))
        fig_err.add_trace(go.Scatter(
            x=list(mc_df["prob"])+list(mc_df["prob"].iloc[::-1]),
            y=list(mc_df["mean_e"]+mc_df["std_e"])+list((mc_df["mean_e"]-mc_df["std_e"]).iloc[::-1]),
            fill="toself", fillcolor="rgba(37,99,235,0.18)",
            line=dict(color="rgba(255,255,255,0)"), name="Mean ± 1σ"))
        fig_err.add_trace(go.Scatter(x=mc_df["prob"], y=mc_df["mean_e"],
            mode="markers+lines",
            marker=dict(size=11,color=C["primary"],line=dict(color="white",width=2)),
            line=dict(color=C["primary"],width=2.5), name="Mean"))
        fig_err.update_layout(title=f"<b>Energy distribution</b> [{ul}]",
            xaxis_title="Request-stop probability", yaxis_title=f"Energy [{ul}]",
            height=400, paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(t=60,b=50,l=60,r=20),
            font=dict(family="Inter, sans-serif",size=12), yaxis=dict(gridcolor=C["light"]),
            legend=dict(orientation="h",yanchor="bottom",y=1.02))
        st.plotly_chart(fig_err, use_container_width=True, key="err_tab4")

    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(
        x=list(mc_df["prob"])+list(mc_df["prob"].iloc[::-1]),
        y=list(mc_df["max_t"]/60)+list(mc_df["min_t"].iloc[::-1]/60),
        fill="toself", fillcolor=C["bg_orange"],
        line=dict(color="rgba(255,255,255,0)"), name="Min–Max range"))
    fig_t.add_trace(go.Scatter(x=mc_df["prob"], y=mc_df["mean_t"]/60,
        mode="markers+lines",
        marker=dict(size=11,color=C["accent"],line=dict(color="white",width=2)),
        line=dict(color=C["accent"],width=2.5), name="Mean time"))
    fig_t.update_layout(title="<b>Journey time vs. stopping policy</b>",
        xaxis_title="Request-stop probability", yaxis_title="Journey time [min]",
        height=360, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(t=60,b=50,l=60,r=20),
        font=dict(family="Inter, sans-serif",size=12), yaxis=dict(gridcolor=C["light"]),
        legend=dict(orientation="h",yanchor="bottom",y=1.02))
    st.plotly_chart(fig_t, use_container_width=True, key="time_tab4")

    with st.expander("📊 Energy–time trade-off"):
        fig_et = go.Figure(go.Scatter(
            x=mc_df["mean_t"]/60, y=mc_df["mean_e"], mode="markers+text",
            marker=dict(size=mc_df["savings"].clip(lower=0)*3+14,
                        color=mc_df["savings"], colorscale="RdYlGn", showscale=True,
                        colorbar=dict(title=f"Savings [{ul}]"),
                        line=dict(color="white",width=1.5)),
            text=mc_df["prob"], textposition="top center",
            hovertemplate="<b>%{text}</b><br>Time: %{x:.1f} min<br>"
                           "Energy: %{y:.2f} "+ul+"<extra></extra>"))
        fig_et.update_layout(title="<b>Energy–time trade-off</b> by stopping policy",
            xaxis_title="Mean journey time [min]", yaxis_title=f"Mean energy [{ul}]",
            height=420, paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(t=60,b=50,l=60,r=20),
            font=dict(family="Inter, sans-serif",size=12))
        st.plotly_chart(fig_et, use_container_width=True, key="tradeoff_tab4")
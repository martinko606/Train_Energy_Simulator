"""
Monte Carlo Fleet Simulator — Integrated Edition
================================================
Merges:
  • RailMLTrackProfileBuilder  (code 1) — graph-based railML 3.x parser with
    gradient / curve / speed / electrification / recuperation awareness
  • TrackProfile + TrainSimulator (code 2 enhanced) — physics engine with full
    electrification checks, recuperation, and history recording
  • Streamlit dashboard         (code 2) — MC analysis, representative-run plots,
    timetable Dijkstra journey search
"""

from __future__ import annotations

import glob
import heapq
import itertools
import math
import random
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import zipfile, tempfile, os

# ============================================================
#  VEHICLE FLEET
# ============================================================
PREDEFINED_VEHICLES: dict[str, dict] = {
    "EDITA": {
        "traction": "DIESEL", "mass": 22_000, "length": 15,
        "power": 152, "aux_power": 20, "accel": 0.5, "decel": 0.8,
        "efficiency": 0.30,
    },
    "EDITA+Btax": {
        "traction": "DIESEL", "mass": 42_000, "length": 30,
        "power": 152, "aux_power": 25, "accel": 0.4, "decel": 0.8,
        "efficiency": 0.30,
    },
    "RegioNova (Class 814)": {
        "traction": "DIESEL", "mass": 80_000, "length": 44,
        "power": 485, "aux_power": 20, "accel": 0.5, "decel": 0.8,
        "efficiency": 0.32,
    },
    "Stadler RS1 (Class 840)": {
        "traction": "DIESEL", "mass": 50_000, "length": 25.5,
        "power": 514, "aux_power": 25, "accel": 0.8, "decel": 0.9,
        "efficiency": 0.38,
    },
    "CityElefant (Class 471)": {
        "traction": "ELECTRIC", "mass": 155_000, "length": 79,
        "power": 2_000, "aux_power": 80, "accel": 0.8, "decel": 0.9,
        "efficiency": 0.85,
    },
}

# ============================================================
#  STREAMLIT PAGE CONFIG & SESSION STATE
# ============================================================
st.set_page_config(page_title="Monte Carlo Fleet Simulator", layout="wide")

for _k in ("mc_results", "rep_results", "journey_results"):
    if _k not in st.session_state:
        st.session_state[_k] = None


# ============================================================
#  RAILML TRACK PROFILE BUILDER  (from code 1 — unchanged API)
# ============================================================
class RailMLTrackProfileBuilder:
    """
    Builds a direction-specific track profile between two graph nodes by
    parsing railML 3.x infrastructure:
      • gradient  (permille, direction-aware)
      • curves    (radius, m)
      • speed     (km/h)
      • stopping points  (X = mandatory, R = request)
      • electrification  (NONE / "VoltageV_FreqHz")
      • recuperation     (0 / 1)
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.root: ET.Element | None = None
        self.graph: dict = {}          # node_id -> list[edge_dict]
        self.nodes: dict = {}          # node_id -> metadata
        self.electrification_systems: dict = {}
        self.electrification_map: dict = {}   # netElement id -> electrification label
        self.speed_map: dict = {}             # netElement id -> speed [km/h]
        self.gradient_map: dict = {}          # netElement id -> gradient [permille]
        self.curve_map: dict = {}             # netElement id -> curve radius [m] | None

        self._load_xml()
        self._parse_electrification()
        self._parse_speed_profiles()
        self._parse_gradients_and_curves()
        self._build_graph()

    # ----------------------------------------------------------
    #  XML helpers
    # ----------------------------------------------------------
    def _load_xml(self):
        tree = ET.parse(self.filepath)
        root = tree.getroot()
        for elem in root.iter():
            if "}" in elem.tag:
                elem.tag = elem.tag.split("}", 1)[1]
        self.root = root

    def _parse_electrification(self):
        for es in self.root.iter("electrificationSystem"):
            es_id = es.get("id")
            voltage = es.get("voltage")
            freq = es.get("frequency")
            if es_id:
                label = "NONE" if voltage == "0" else f"{voltage}V_{freq}Hz"
                self.electrification_systems[es_id] = label

        for ne in self.root.iter("netElement"):
            ne_id = ne.get("id")
            es_ref = ne.get("electrificationRef")
            if ne_id and es_ref:
                self.electrification_map[ne_id] = self.electrification_systems.get(es_ref, "UNKNOWN")

    def _parse_speed_profiles(self):
        sp_speed: dict = {}
        for sp in self.root.iter("speedProfile"):
            sp_id = sp.get("id")
            vmax = sp.get("maxSpeed")
            if sp_id and vmax:
                sp_speed[sp_id] = float(vmax)

        for ne in self.root.iter("netElement"):
            ne_id = ne.get("id")
            sp_ref = ne.get("speedProfileRef")
            if ne_id and sp_ref and sp_ref in sp_speed:
                self.speed_map[ne_id] = sp_speed[sp_ref]

    def _parse_gradients_and_curves(self):
        for ne in self.root.iter("netElement"):
            ne_id = ne.get("id")
            if not ne_id:
                continue

            grad_val = 0.0
            grad_elem = ne.find("gradient")
            if grad_elem is not None:
                v = grad_elem.get("value")
                if v is not None:
                    try:
                        grad_val = float(v)
                    except Exception:
                        grad_val = 0.0
            self.gradient_map[ne_id] = grad_val

            curve_r = None
            curve_elem = ne.find("curve")
            if curve_elem is not None:
                r = curve_elem.get("radius")
                if r is not None:
                    try:
                        curve_r = float(r)
                    except Exception:
                        curve_r = None
            self.curve_map[ne_id] = curve_r

    # ----------------------------------------------------------
    #  Graph construction
    # ----------------------------------------------------------
    @staticmethod
    def _segment_length(u, v) -> float:
        if len(u) == 3 and len(v) == 3:
            _, x1, y1 = u
            _, x2, y2 = v
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 1000.0
        if len(u) == 2 and len(v) == 2:
            _, m1 = u
            _, m2 = v
            return abs(m2 - m1)
        return 0.0

    def _add_edge(self, u, v, length_m: float, ne_id: str):
        electr = self.electrification_map.get(ne_id, "NONE")
        for a, b in ((u, v), (v, u)):
            self.graph.setdefault(a, []).append(
                {
                    "to": b,
                    "length_m": length_m,
                    "netElement": ne_id,
                    "gradient_permille": self.gradient_map.get(ne_id, 0.0),
                    "curve_radius": self.curve_map.get(ne_id),
                    "speed_kmh": self.speed_map.get(ne_id, 80.0),
                    "electrification": electr,
                    "recuperation": 0 if electr == "NONE" else 1,
                }
            )

    def _build_graph(self):
        for ne in self.root.iter("netElement"):
            ne_id = ne.get("id")
            if not ne_id:
                continue

            endpoints: list = []
            for aps in ne.findall("associatedPositioningSystem"):
                ps_ref = aps.get("positioningSystemRef")
                for ic in aps.findall("intrinsicCoordinate"):
                    geo = ic.find("geometricCoordinate")
                    lin = ic.find("linearCoordinate")

                    if geo is not None:
                        x = float(geo.get("x"))
                        y = float(geo.get("y"))
                        nid = (ps_ref, round(x, 8), round(y, 8))
                        endpoints.append(nid)
                        if nid not in self.nodes:
                            self.nodes[nid] = {
                                "name": f"{ne_id}@{ic.get('intrinsicCoord')}",
                                "type": "NODE",
                                "coord": (x, y),
                            }
                    elif lin is not None:
                        m = float(lin.get("measure"))
                        nid = (ps_ref, round(m, 3))
                        endpoints.append(nid)
                        if nid not in self.nodes:
                            self.nodes[nid] = {
                                "name": f"{ne_id}@{ic.get('intrinsicCoord')}",
                                "type": "NODE",
                                "coord": m,
                            }

            if len(endpoints) == 2:
                u, v = endpoints
                self._add_edge(u, v, self._segment_length(u, v), ne_id)

    # ----------------------------------------------------------
    #  Public API
    # ----------------------------------------------------------
    def get_stopping_points(self) -> list[tuple]:
        """
        Returns [(node_id, display_name, stop_type), ...]
        stop_type ∈ {"X", "R"}
        """
        stops: list = []
        for op in self.root.iter("operationalPoint"):
            op_id = op.get("id")
            name = op.get("name", op_id)
            stop_type = op.get("stopType", "X")
            if stop_type not in ("X", "R"):
                continue

            geo = op.find(".//geometricCoordinate")
            if geo is None:
                continue
            x, y = float(geo.get("x")), float(geo.get("y"))

            best_node, best_d = None, float("inf")
            for nid, meta in self.nodes.items():
                if len(nid) != 3:
                    continue
                _, nx, ny = nid
                d = (nx - x) ** 2 + (ny - y) ** 2
                if d < best_d:
                    best_d = d
                    best_node = nid

            if best_node is not None:
                stops.append((best_node, name, stop_type))
        return stops

    def dijkstra(self, start_node, end_node) -> tuple[float | None, list]:
        pq: list = []
        tb = itertools.count()
        heapq.heappush(pq, (0.0, next(tb), start_node, []))
        dist: dict = {start_node: 0.0}

        while pq:
            cur_d, _, node, path = heapq.heappop(pq)
            if node == end_node:
                return cur_d, path
            if cur_d > dist.get(node, float("inf")):
                continue
            for edge in self.graph.get(node, []):
                nxt = edge["to"]
                nd = cur_d + edge["length_m"]
                if nd < dist.get(nxt, float("inf")):
                    dist[nxt] = nd
                    heapq.heappush(pq, (nd, next(tb), nxt, path + [(node, edge)]))
        return None, []

    def build_profile(self, start_node, end_node, direction: int = 1) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
          Poloha, Dopravní bod, Zastavení, Rychlost, Sklon,
          Electrification, Recuperation, CurveRadius
        sorted descending by Poloha (high → low km).
        """
        _, path = self.dijkstra(start_node, end_node)
        if not path:
            return pd.DataFrame()

        stop_map = {
            nid: (nm, st) for nid, nm, st in self.get_stopping_points()
        }

        rows: list[dict] = []
        cum_km = 0.0
        node_index = 0

        if direction < 0:
            rev: list = []
            last_to = path[-1][1]["to"]
            for from_node, edge in reversed(path):
                new_edge = dict(edge)
                new_edge["to"] = from_node
                rev.append((last_to, new_edge))
                last_to = from_node
            path = rev

        for from_node, edge in path:
            grad_permille = edge["gradient_permille"]
            if direction < 0:
                grad_permille = -grad_permille

            stop_name, stop_type = stop_map.get(from_node, ("", ""))
            if not stop_name:
                stop_name = self.nodes.get(from_node, {}).get("name", f"Node_{node_index}")

            rows.append(
                {
                    "Poloha": cum_km,
                    "Dopravní bod": stop_name,
                    "Zastavení": stop_type,
                    "Rychlost": edge["speed_kmh"],
                    "Sklon": grad_permille,
                    "Electrification": edge["electrification"],
                    "Recuperation": edge["recuperation"],
                    "CurveRadius": edge["curve_radius"],
                }
            )
            cum_km += edge["length_m"] / 1000.0
            node_index += 1

        # Final node
        last_edge = path[-1][1]
        grad_last = -last_edge["gradient_permille"] if direction < 0 else last_edge["gradient_permille"]
        final_name, final_type = stop_map.get(end_node, ("", ""))
        if not final_name:
            final_name = self.nodes.get(end_node, {}).get("name", f"Node_{node_index}")

        rows.append(
            {
                "Poloha": cum_km,
                "Dopravní bod": final_name,
                "Zastavení": final_type,
                "Rychlost": last_edge["speed_kmh"],
                "Sklon": grad_last,
                "Electrification": last_edge["electrification"],
                "Recuperation": last_edge["recuperation"],
                "CurveRadius": last_edge["curve_radius"],
            }
        )

        df = pd.DataFrame(rows)
        # Normalise: sort descending so high-km = row 0
        df = df.sort_values("Poloha", ascending=False).reset_index(drop=True)
        return df


# ============================================================
#  TIMETABLE UTILITIES  (from code 2 — unchanged)
# ============================================================
def parse_time_to_seconds(t_str: str | None) -> int | None:
    if not t_str:
        return None
    try:
        parts = t_str.split(":")
        h = int(parts[0])
        m = int(parts[1])
        s = int(parts[2]) if len(parts) > 2 else 0
        return h * 3600 + m * 60 + s
    except Exception:
        return None


def parse_duration_to_seconds(d_str: str | None) -> int:
    if not d_str:
        return 0
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", d_str)
    if match:
        h = int(match.group(1) or 0)
        m = int(match.group(2) or 0)
        s = int(match.group(3) or 0)
        return h * 3600 + m * 60 + s
    return 0


def format_time(sec: int | float | None, *, is_duration: bool = False) -> str:
    if sec is None:
        return ""
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if is_duration:
        return f"{h:02d}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"
    return f"{h % 24:02d}:{m:02d}:{s:02d}"


class TimetableGraph:
    def __init__(self):
        self.edges: dict = {}
        self.connections: dict = {}
        self.ocp_id_to_name: dict = {}
        self.ocp_name_to_id: dict = {}

    def add_edge(self, from_ocp, to_ocp, tp_id, dep_sec, arr_sec):
        self.edges.setdefault(from_ocp, []).append(
            {"to": to_ocp, "trainPart": tp_id, "dep": dep_sec, "arr": arr_sec, "duration": arr_sec - dep_sec}
        )

    def add_connection(self, ocp, from_tp, to_tp, min_time_sec):
        self.connections.setdefault((ocp, from_tp), []).append(
            {"to_tp": to_tp, "min_time": min_time_sec}
        )


@st.cache_data
def parse_railml_timetable(filepath: str) -> TimetableGraph:
    graph = TimetableGraph()
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()
        for elem in root.iter():
            if "}" in elem.tag:
                elem.tag = elem.tag.split("}", 1)[1]

        for ocp in root.iter("ocp"):
            ocp_id = ocp.get("id")
            ocp_name = ocp.get("name", ocp_id)
            if ocp_id:
                graph.ocp_id_to_name[ocp_id] = ocp_name
                graph.ocp_name_to_id[ocp_name] = ocp_id

        for tp in root.iter("trainPart"):
            tp_id = tp.get("id")
            ocp_tts = list(tp.findall(".//ocpTT"))
            for i in range(len(ocp_tts) - 1):
                cur, nxt = ocp_tts[i], ocp_tts[i + 1]
                ocp_a, ocp_b = cur.get("ocpRef"), nxt.get("ocpRef")
                ta, tb_ = cur.find("times"), nxt.find("times")
                if ta is not None and tb_ is not None:
                    dep = parse_time_to_seconds(ta.get("departure"))
                    arr = parse_time_to_seconds(tb_.get("arrival"))
                    if dep is not None and arr is not None:
                        if arr < dep:
                            arr += 86400
                        graph.add_edge(ocp_a, ocp_b, tp_id, dep, arr)
                for conn in cur.findall(".//connection"):
                    to_tp = conn.get("trainPartRef") or conn.get("trainRef")
                    min_t = parse_duration_to_seconds(conn.get("minConnTime"))
                    if to_tp:
                        graph.add_connection(ocp_a, tp_id, to_tp, min_t)
    except Exception as e:
        st.warning(f"Timetable Graph error: {e}")
    return graph


def find_journey_dijkstra(
    graph: TimetableGraph,
    start_ocp: str,
    end_ocp: str,
    departure_time: str,
) -> dict | None:
    tb = itertools.count()
    start_sec = parse_time_to_seconds(departure_time)
    if start_sec is None:
        raise ValueError("Invalid departure time. Use HH:MM or HH:MM:SS.")

    start_id = graph.ocp_name_to_id.get(start_ocp, start_ocp)
    end_id = graph.ocp_name_to_id.get(end_ocp, end_ocp)
    if start_id not in graph.edges:
        return None

    pq: list = []
    heapq.heappush(pq, (start_sec, next(tb), start_id, None, []))
    visited: dict = {}

    while pq:
        cur_time, _, cur_ocp, cur_tp, path = heapq.heappop(pq)
        if cur_ocp == end_id:
            dur = cur_time - start_sec
            fmt_path = [
                {
                    "from": graph.ocp_id_to_name.get(s["from"], s["from"]),
                    "to": graph.ocp_id_to_name.get(s["to"], s["to"]),
                    "trainPart": s["trainPart"],
                    "dep": format_time(s["dep"]),
                    "arr": format_time(s["arr"]),
                }
                for s in path
            ]
            return {"total_time": format_time(dur, is_duration=True), "path": fmt_path}

        state = (cur_ocp, cur_tp)
        if state in visited and visited[state] <= cur_time:
            continue
        visited[state] = cur_time

        for edge in graph.edges.get(cur_ocp, []):
            nxt_tp = edge["trainPart"]
            e_dep, e_arr = edge["dep"], edge["arr"]
            while e_dep < cur_time:
                e_dep += 86400
                e_arr += 86400

            if cur_tp is not None and cur_tp != nxt_tp:
                allowed, min_transfer = True, 0
                conn_key = (cur_ocp, cur_tp)
                if conn_key in graph.connections:
                    allowed = False
                    for conn in graph.connections[conn_key]:
                        if conn["to_tp"] == nxt_tp:
                            allowed = True
                            min_transfer = conn["min_time"]
                            break
                if not allowed:
                    continue
                while cur_time + min_transfer > e_dep:
                    e_dep += 86400
                    e_arr += 86400

            heapq.heappush(
                pq,
                (
                    e_arr,
                    next(tb),
                    edge["to"],
                    nxt_tp,
                    path + [{"from": cur_ocp, "to": edge["to"], "trainPart": nxt_tp, "dep": e_dep, "arr": e_arr}],
                ),
            )
    return None


# ============================================================
#  CACHED BUILDER (resource-level cache so the graph lives
#  across reruns without re-parsing the XML every time)
# ============================================================
@st.cache_resource
def get_railml_builder(filepath: str) -> RailMLTrackProfileBuilder:
    return RailMLTrackProfileBuilder(filepath)


@st.cache_data
def get_stopping_point_labels(filepath: str) -> list[tuple]:
    """
    Returns [(display_label, node_id, stop_type), ...]
    The display_label is used in selectboxes.
    """
    builder = get_railml_builder(filepath)
    raw = builder.get_stopping_points()          # (node_id, name, stop_type)
    seen: dict[str, int] = {}
    result: list = []
    for node_id, name, stop_type in raw:
        key = name
        seen[key] = seen.get(key, 0) + 1
        label = f"{name} [{stop_type}]" if seen[key] == 1 else f"{name} [{stop_type}] #{seen[key]}"
        result.append((label, node_id, stop_type))
    return result


# ============================================================
#  EXCEL / LEGACY DATA LOADERS  (code 2 path, used for .xlsx)
# ============================================================
@st.cache_data
def get_excel_dataframe(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()
    df["Zastavení"] = df["Zastavení"].fillna("")
    df["Poloha"] = pd.to_numeric(df["Poloha"], errors="coerce")
    df["Rychlost"] = pd.to_numeric(df["Rychlost"], errors="coerce")
    df["Sklon"] = pd.to_numeric(df["Sklon"], errors="coerce")
    df = df.dropna(subset=["Poloha"])
    df["Poloha"] = df["Poloha"].apply(lambda x: x / 1000.0 if x > 1000 else x)
    # Ensure optional columns exist
    for col in ("Electrification", "Recuperation", "CurveRadius"):
        if col not in df.columns:
            df[col] = "NONE" if col == "Electrification" else 0
    return df


# ============================================================
#  TRACK PROFILE  (merged — adds electrification / recuperation)
# ============================================================
class TrackProfile:
    """
    Consumes a DataFrame produced either by RailMLTrackProfileBuilder.build_profile()
    or by get_excel_dataframe().  Handles columns:
      Poloha, Dopravní bod, Zastavení, Rychlost, Sklon,
      Electrification, Recuperation, CurveRadius
    """

    def __init__(self, df_raw: pd.DataFrame, is_forward_direction: bool):
        self.is_forward = is_forward_direction
        self.df_raw = self._clean(df_raw)
        self.segments = self._build_segments()
        self.stations = self._extract_stations()
        self.station_dict = {s["name"]: s["km"] for s in self.stations}

    # ----------------------------------------------------------
    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_values("Poloha", ascending=False).reset_index(drop=True)
        df["Rychlost"] = df["Rychlost"].ffill().bfill()
        df["Sklon"] = df["Sklon"].ffill().bfill().fillna(0)
        for col, default in (
            ("Electrification", "NONE"),
            ("Recuperation", 0),
            ("CurveRadius", None),
        ):
            if col not in df.columns:
                df[col] = default
        return df

    def _build_segments(self) -> list[dict]:
        segs: list[dict] = []
        for i in range(len(self.df_raw) - 1):
            row = self.df_raw.loc[i]
            km_high = float(row["Poloha"])
            km_low = float(self.df_raw.loc[i + 1, "Poloha"])
            v_limit = float(row["Rychlost"]) / 3.6          # → m/s
            grad = float(row["Sklon"]) / 1000.0              # permille → fraction
            if not self.is_forward:
                grad *= -1
            segs.append(
                {
                    "km_high": km_high,
                    "km_low": km_low,
                    "v_limit": v_limit,
                    "grad": grad,
                    "electrification": str(row["Electrification"]),
                    "recuperation": int(row["Recuperation"]),
                    "curve_radius": row.get("CurveRadius"),
                }
            )
        return segs

    def _extract_stations(self) -> list[dict]:
        stations: list[dict] = []
        for _, row in self.df_raw.iterrows():
            name = str(row.get("Dopravní bod", "")).strip()
            stype = str(row["Zastavení"]).strip().upper()
            if name and name != "nan" and stype in ("X", "R"):
                stations.append({"name": name, "km": row["Poloha"], "type": stype})
        return stations

    def get_segment_properties(
        self, front_km: float, rear_km: float
    ) -> tuple[float, float, str, int]:
        """
        Returns (v_limit_m_s, gradient_fraction, electrification_label, recuperation_flag)
        for the span [min(front,rear) … max(front,rear)].
        """
        span_min, span_max = min(front_km, rear_km), max(front_km, rear_km)
        limits: list[float] = []
        grad, electr, recup = 0.0, "NONE", 0
        eps = 1e-6
        for seg in self.segments:
            if span_min <= seg["km_high"] + eps and span_max >= seg["km_low"] - eps:
                limits.append(seg["v_limit"])
            if seg["km_low"] - eps <= front_km <= seg["km_high"] + eps:
                grad = seg["grad"]
                electr = seg["electrification"]
                recup = seg["recuperation"]
        return (min(limits) if limits else 0.0), grad, electr, recup


# ============================================================
#  TRAIN SIMULATOR  (merged — electrification checks + recup
#  from code 1;  _build_events + history recording from code 2)
# ============================================================
class TrainSimulator:
    def __init__(
        self,
        mass_kg: float,
        length_m: float,
        max_power_kw: float,
        aux_power_kw: float,
        max_accel: float,
        max_decel: float,
        traction_type: str,
        efficiency: float,
    ):
        self.mass_kg = mass_kg
        self.eff_mass = mass_kg * 1.08          # rotational inertia factor
        self.A, self.B, self.C = 1500.0, 30.0, 4.0   # Davis resistance coefficients
        self.traction_type = traction_type
        self.traction_efficiency = efficiency if traction_type == "ELECTRIC" else 0.85
        self.regen_efficiency = 0.75 if traction_type == "ELECTRIC" else 0.0
        self.aux_power_w = aux_power_kw * 1000.0
        self.max_power_w = max_power_kw * 1000.0
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.train_length_m = length_m

    # ----------------------------------------------------------
    def get_resistance(self, v_m_s: float) -> float:
        return self.A + self.B * v_m_s + self.C * v_m_s ** 2

    # ----------------------------------------------------------
    def _build_events(
        self,
        track: TrackProfile,
        start_km: float,
        end_km: float,
        stop_mode: str,
        stop_prob: float,
        direction: int,
    ) -> tuple[list[dict], list[float], list[str]]:
        km_min, km_max = min(start_km, end_km), max(start_km, end_km)
        stops_km: list[float] = []
        stops_names: list[str] = []
        events: list[dict] = []

        for station in track.stations:
            km = station["km"]
            if not (km_min <= km <= km_max):
                continue
            if abs(km - start_km) < 0.01:
                continue

            will_stop = False
            if station["type"] == "X":
                will_stop = True
            elif station["type"] == "R":
                if stop_mode == "all":
                    will_stop = True
                elif stop_mode == "random" and random.random() <= stop_prob:
                    will_stop = True

            if will_stop:
                stops_km.append(km)
                stops_names.append(station["name"])
                events.append({"km": km, "target_v": 0.0, "type": "stop"})

        for seg in track.segments:
            boundary = seg["km_high"] if direction == -1 else seg["km_low"]
            if km_min - 0.01 <= boundary <= km_max + 0.01:
                events.append({"km": boundary, "target_v": seg["v_limit"], "type": "limit"})

        return events, stops_km, stops_names

    # ----------------------------------------------------------
    def run_simulation(
        self,
        track: TrackProfile,
        start_km: float,
        end_km: float,
        stop_mode: str = "all",
        stop_prob: float = 1.0,
        dwell_time: float = 30.0,
        record_history: bool = False,
    ) -> tuple[dict | None, list[str], dict]:
        """
        Returns (history_dict_or_None, stops_names, stats_dict)
        stats_dict keys: net_kwh, journey_time_s
        """
        total_dist_m = abs(start_km - end_km) * 1000.0
        direction = -1 if start_km > end_km else 1

        events, stops_km, stops_names = self._build_events(
            track, start_km, end_km, stop_mode, stop_prob, direction
        )

        dt = 1.0
        current_v = 0.0
        distance_covered = 0.0
        total_energy_j = 0.0
        total_regen_j = 0.0
        journey_time_s = 0.0
        g = 9.8186

        history: dict | None = (
            {k: [] for k in ("time_s", "km", "cum_dist_km", "v_actual", "v_limit", "energy_kwh", "regen_kwh")}
            if record_history
            else None
        )

        while distance_covered < total_dist_m:
            current_km = start_km + (distance_covered / 1000.0) * direction
            rear_km = current_km - (self.train_length_m / 1000.0) * direction

            # ── segment properties (code-1 style: includes electr / recup) ──
            v_limit, slope, electr, recup_allowed = track.get_segment_properties(current_km, rear_km)

            # ── electrification guard (from code 1) ──
            if self.traction_type == "ELECTRIC" and electr == "NONE":
                raise RuntimeError(
                    f"Electric train on non-electrified segment at km {current_km:.3f}. "
                    "Choose a diesel vehicle or select an electrified route."
                )

            f_gradient = self.mass_kg * g * slope
            effective_decel = max(0.05, self.max_decel + g * slope)
            max_safe_v = v_limit

            # ── speed-safe braking profile for upcoming events ──
            for event in events:
                tol = 0.05 if event["type"] == "stop" else 1e-4
                is_ahead = (
                    event["km"] <= current_km + tol
                    if direction == -1
                    else event["km"] >= current_km - tol
                )
                overshot = (
                    current_km < event["km"]
                    if direction == -1
                    else current_km > event["km"]
                )
                if not is_ahead:
                    continue
                dist_to_event = (
                    0.0
                    if (event["type"] == "stop" and overshot)
                    else abs(current_km - event["km"]) * 1000.0
                )
                if event["target_v"] < current_v:
                    safe_v = math.sqrt(
                        max(0.0, event["target_v"] ** 2 + 2.0 * effective_decel * dist_to_event)
                    )
                    max_safe_v = min(max_safe_v, safe_v)

            mech_power = 0.0
            regen_power = 0.0

            # ── optional history snapshot (before physics step) ──
            if record_history:
                lim_front, *_ = track.get_segment_properties(current_km, current_km)
                history["time_s"].append(journey_time_s)
                history["km"].append(current_km)
                history["cum_dist_km"].append(distance_covered / 1000.0)
                history["v_actual"].append(current_v * 3.6)
                history["v_limit"].append(lim_front * 3.6)
                history["energy_kwh"].append(total_energy_j / 3_600_000.0)
                history["regen_kwh"].append(total_regen_j / 3_600_000.0)

            # ── physics step ──
            v_eff = max(current_v, 0.5)   # avoid division by zero at standstill

            if current_v > max_safe_v + 1e-4:
                # BRAKING
                f_resist = self.get_resistance(current_v)
                natural_decel = (f_resist + f_gradient) / self.eff_mass
                required_decel = (current_v - max_safe_v) / dt
                brake = max(0.0, min(self.max_decel, required_decel - natural_decel))
                actual_decel = brake + natural_decel
                # Recuperation only when electric AND segment allows it (code 1 logic)
                if self.traction_type == "ELECTRIC" and recup_allowed:
                    regen_power = self.eff_mass * brake * current_v * self.regen_efficiency
                current_v = max(0.0, current_v - actual_decel * dt)

            elif current_v < min(v_limit, max_safe_v) - 1e-4:
                # TRACTION
                f_resist = self.get_resistance(current_v)
                desired_force = f_resist + self.eff_mass * self.max_accel + f_gradient
                actual_force = max(0.0, min(desired_force, self.max_power_w / v_eff))
                current_v = max(
                    0.0,
                    min(
                        current_v + (actual_force - f_resist - f_gradient) / self.eff_mass * dt,
                        v_limit,
                        max_safe_v,
                    ),
                )
                mech_power = max(0.0, actual_force * current_v)

            else:
                # CRUISING
                f_resist = self.get_resistance(current_v)
                desired_force = f_resist + f_gradient
                actual_force = max(0.0, min(desired_force, self.max_power_w / v_eff))
                current_v = max(0.0, current_v + (actual_force - f_resist - f_gradient) / self.eff_mass * dt)
                mech_power = max(0.0, actual_force * current_v)

            total_energy_j += (mech_power / self.traction_efficiency + self.aux_power_w) * dt
            total_regen_j += regen_power * dt
            distance_covered += current_v * dt
            journey_time_s += dt

            # ── dwell at stop ──
            if current_v < 0.5 and any(abs(current_km - s) <= 0.05 for s in stops_km):
                current_v = 0.0
                if record_history:
                    # two extra snapshots to show the flat dwell segment
                    for pass_no in range(2):
                        lim_front, *_ = track.get_segment_properties(current_km, current_km)
                        history["time_s"].append(journey_time_s)
                        history["km"].append(current_km)
                        history["cum_dist_km"].append(distance_covered / 1000.0)
                        history["v_actual"].append(0.0)
                        history["v_limit"].append(lim_front * 3.6)
                        history["energy_kwh"].append(total_energy_j / 3_600_000.0)
                        history["regen_kwh"].append(total_regen_j / 3_600_000.0)
                        if pass_no == 0:
                            total_energy_j += self.aux_power_w * dwell_time
                            journey_time_s += dwell_time
                else:
                    total_energy_j += self.aux_power_w * dwell_time
                    journey_time_s += dwell_time

                events = [e for e in events if not (e["type"] == "stop" and abs(e["km"] - current_km) <= 0.05)]
                stops_km = [s for s in stops_km if abs(s - current_km) > 0.05]

        stats = {
            "net_kwh": (total_energy_j - total_regen_j) / 3_600_000.0,
            "journey_time_s": journey_time_s,
        }
        return history, stops_names, stats


# ============================================================
#  STATION LOADER HELPERS
# ============================================================
def load_stations_from_railml(filepath: str) -> list[tuple[str, float, object]]:
    """
    Returns [(display_name, position_km, node_id), ...] sorted by position desc.
    position_km is derived from cumulative graph distance (not a real coordinate).
    """
    labels = get_stopping_point_labels(filepath)   # (label, node_id, stop_type)
    if not labels:
        return []
    builder = get_railml_builder(filepath)
    # We need km positions — build a quick profile from first to last stop
    # to get a positional ordering, then use index as a proxy
    # (Real km only available after build_profile is called with selected nodes)
    result = [(lbl, float(i), nid) for i, (lbl, nid, _) in enumerate(labels)]
    return result


def load_stations_from_excel(filepath: str) -> list[tuple[str, float, None]]:
    df = get_excel_dataframe(filepath)
    if df.empty:
        return []
    stations = []
    for _, row in df.iterrows():
        if str(row["Zastavení"]).strip().upper() in ("X", "R"):
            name = str(row.get("Dopravní bod", "")).strip()
            if name and name != "nan":
                stations.append((name, float(row["Poloha"]), None))
    return sorted(stations, key=lambda x: x[1], reverse=True)


# ============================================================
#  STREAMLIT APP
# ============================================================
st.title("🎲 Monte Carlo Fleet Simulator")
st.markdown(
    "Stochastic energy / time analysis with **railML 3.x graph routing** "
    "(gradient · curvature · electrification · recuperation) + Excel track profiles."
)

# ── 0. FILE LOADER ──────────────────────────────────────────
st.sidebar.header("📂 Data Source")
ZIP_PATH = "railML_export_20251214_20260414.zip"   # ← set this
EXTRACT_DIR = tempfile.mkdtemp(prefix="railml_")

with zipfile.ZipFile(ZIP_PATH) as zf:
    zf.extractall(EXTRACT_DIR)

all_files = sorted(
    f for f in
    glob.glob(f"{EXTRACT_DIR}/**/*.xlsx", recursive=True) +
    glob.glob(f"{EXTRACT_DIR}/**/*.xml",  recursive=True) +
    glob.glob(f"{EXTRACT_DIR}/**/*.railml", recursive=True)
    if not os.path.basename(f).startswith("~$")
)

if not all_files:
    st.sidebar.error("No Excel (.xlsx) or railML (.xml/.railml) files found in the working directory.")
    st.stop()

selected_file = st.sidebar.selectbox("Select Track / Timetable File", all_files)
is_railml = selected_file.lower().endswith((".xml", ".railml"))

# ── Station loading — railML vs Excel ───────────────────────
if is_railml:
    st.sidebar.info("railML 3.x detected — using graph-based RailMLTrackProfileBuilder.")
    with st.spinner("Parsing railML infrastructure graph…"):
        stop_labels = get_stopping_point_labels(selected_file)   # (label, node_id, stop_type)

    if not stop_labels:
        st.error(
            "No operational stopping points found in this railML file. "
            "Ensure `<operationalPoint>` elements with `<geometricCoordinate>` are present."
        )
        st.stop()

    label_list = [x[0] for x in stop_labels]
    label_to_node = {x[0]: x[1] for x in stop_labels}
    label_to_type = {x[0]: x[2] for x in stop_labels}
    station_names = label_list

else:
    stations_raw = load_stations_from_excel(selected_file)
    if not stations_raw:
        st.error(f"No stations (Zastavení = X/R) found in {selected_file}.")
        st.stop()
    station_names = [s[0] for s in stations_raw]
    station_dict_excel = {s[0]: s[1] for s in stations_raw}

# ── 1. VEHICLE ───────────────────────────────────────────────
st.sidebar.header("1. Vehicle")
vehicle_choice = st.sidebar.selectbox("Vehicle Profile", ["Custom"] + list(PREDEFINED_VEHICLES.keys()))

if vehicle_choice == "Custom":
    traction = st.sidebar.selectbox("Traction Type", ["DIESEL", "ELECTRIC"])
    mass = st.sidebar.number_input("Train Mass (kg)", value=100_000, step=10_000)
    length = st.sidebar.number_input("Train Length (m)", value=40, step=10)
    power = st.sidebar.number_input("Max Power (kW)", value=600, step=50)
    aux_power = st.sidebar.number_input("Auxiliary Power (kW)", value=40, step=5)
    accel = st.sidebar.slider("Max Acceleration (m/s²)", 0.2, 1.2, 0.6, 0.1)
    decel = st.sidebar.slider("Max Braking (m/s²)", 0.4, 1.5, 0.8, 0.1)
    efficiency = st.sidebar.slider("Efficiency (%)", 15, 95, 35) / 100.0
    diesel_density = st.sidebar.number_input("Diesel Density (kWh/L)", value=10.0, step=0.1)
else:
    p = PREDEFINED_VEHICLES[vehicle_choice]
    traction, mass, length = p["traction"], p["mass"], p["length"]
    power, aux_power = p["power"], p["aux_power"]
    accel, decel, efficiency = p["accel"], p["decel"], p["efficiency"]
    diesel_density = 10.0

# ── 2. SHIFT BUILDER ─────────────────────────────────────────
st.sidebar.header("2. Shift Configuration")
mc_start_label = st.sidebar.selectbox("Terminal A", station_names, index=0)
mc_end_label = st.sidebar.selectbox("Terminal B", station_names, index=len(station_names) - 1)

trip_pattern = st.sidebar.radio("Trip Pattern", ["Single Direction (A ➔ B)", "Round Trip (A ➔ B ➔ A)"])
num_cycles = st.sidebar.number_input("Number of Cycles", min_value=1, max_value=20, value=1)

st.sidebar.header("3. Stochastic Settings")
mc_runs = st.sidebar.number_input("N (Runs per Probability)", min_value=10, max_value=1000, value=100, step=10)
mc_dwell = st.sidebar.number_input("Dwell Time (s)", value=30, step=5)

st.sidebar.header("4. Representative Run")
plot_dir = st.sidebar.radio("Plot Direction", [f"{mc_start_label} ➔ {mc_end_label}", f"{mc_end_label} ➔ {mc_start_label}"])
plot_prob = st.sidebar.slider("Stop Probability for Plot", 0.0, 1.0, 0.4, 0.1)
plot_x_axis = st.sidebar.radio("Plot X-Axis", ["Distance (km)", "Time (MM:SS)"])

# ── Helper: build profile DataFrame from selected terminals ──
def resolve_profile(start_label: str, end_label: str) -> tuple[pd.DataFrame, float, float]:
    """
    Returns (df_profile, start_km, end_km).
    For railML: uses RailMLTrackProfileBuilder.build_profile().
    For Excel: slices the preloaded DataFrame.
    """
    if is_railml:
        builder = get_railml_builder(selected_file)
        start_node = label_to_node[start_label]
        end_node = label_to_node[end_label]
        # direction = 1 (forward in file order); build_profile handles reversal
        df = builder.build_profile(start_node, end_node, direction=1)
        if df.empty:
            raise RuntimeError(
                f"No path found between '{start_label}' and '{end_label}' in the railML graph. "
                "Verify that both stopping points are connected in the infrastructure."
            )
        start_km = float(df["Poloha"].max())
        end_km = float(df["Poloha"].min())
    else:
        df = get_excel_dataframe(selected_file)
        start_km = station_dict_excel[start_label]
        end_km = station_dict_excel[end_label]
    return df, start_km, end_km


def make_track(df: pd.DataFrame, s_km: float, e_km: float) -> TrackProfile:
    return TrackProfile(df, is_forward_direction=(s_km > e_km))


def unit_fn(stats: dict) -> float:
    if traction == "DIESEL":
        return stats["net_kwh"] / (diesel_density * efficiency)
    return stats["net_kwh"]


unit_label = "Liters" if traction == "DIESEL" else "kWh"

# ── BUTTONS ─────────────────────────────────────────────────
c_mc, c_plot = st.sidebar.columns(2)
run_mc = c_mc.button("🎲 Run MC", type="primary", use_container_width=True)
run_plot = c_plot.button("📈 Plot Run", use_container_width=True)

# ============================================================
#  MONTE CARLO EXECUTION
# ============================================================
if run_mc:
    if mc_start_label == mc_end_label:
        st.sidebar.error("Start and End must differ!")
        st.stop()

    with st.spinner(f"Running Monte Carlo (N={mc_runs})…"):
        try:
            df_ab, s1, e1 = resolve_profile(mc_start_label, mc_end_label)
            df_ba, s2, e2 = resolve_profile(mc_end_label, mc_start_label)

            track_1 = make_track(df_ab, s1, e1)
            track_2 = make_track(df_ba, s2, e2)

            sim = TrainSimulator(mass, length, power, aux_power, accel, decel, traction, efficiency)

            n_fwd = num_cycles
            n_rev = num_cycles if "Round" in trip_pattern else 0
            total_legs = n_fwd + n_rev

            # ─ Baselines ─
            _, w_names_1, w_stats_1 = sim.run_simulation(track_1, s1, e1, "all", 1.0, mc_dwell)
            _, b_names_1, b_stats_1 = sim.run_simulation(track_1, s1, e1, "none", 0.0, mc_dwell)

            if n_rev > 0:
                _, w_names_2, w_stats_2 = sim.run_simulation(track_2, s2, e2, "all", 1.0, mc_dwell)
                _, b_names_2, b_stats_2 = sim.run_simulation(track_2, s2, e2, "none", 0.0, mc_dwell)
            else:
                w_names_2, w_stats_2 = [], {"journey_time_s": 0, "net_kwh": 0}
                b_names_2, b_stats_2 = [], {"journey_time_s": 0, "net_kwh": 0}

            base_worst = unit_fn(w_stats_1) * n_fwd + unit_fn(w_stats_2) * n_rev
            base_best = unit_fn(b_stats_1) * n_fwd + unit_fn(b_stats_2) * n_rev
            base_worst_time = w_stats_1["journey_time_s"] * n_fwd + w_stats_2["journey_time_s"] * n_rev
            base_best_time = b_stats_1["journey_time_s"] * n_fwd + b_stats_2["journey_time_s"] * n_rev
            base_worst_stops = len(w_names_1) * n_fwd + len(w_names_2) * n_rev
            base_best_stops = len(b_names_1) * n_fwd + len(b_names_2) * n_rev
            mand_per_leg = base_best_stops / total_legs
            req_per_leg = (base_worst_stops - base_best_stops) / total_legs

            # ─ Stochastic sweep ─
            rows_mc: list[dict] = []
            for p in (0.8, 0.6, 0.4, 0.2):
                t_sum = e_sum = s_sum = 0.0
                for _ in range(int(mc_runs)):
                    for _ in range(n_fwd):
                        _, rn, rs = sim.run_simulation(track_1, s1, e1, "random", p, mc_dwell)
                        t_sum += rs["journey_time_s"]
                        e_sum += unit_fn(rs)
                        s_sum += len(rn)
                    for _ in range(n_rev):
                        _, rn, rs = sim.run_simulation(track_2, s2, e2, "random", p, mc_dwell)
                        t_sum += rs["journey_time_s"]
                        e_sum += unit_fn(rs)
                        s_sum += len(rn)
                rows_mc.append(
                    {
                        "Probability": f"{int(p * 100)}%",
                        "Prob_Num": int(p * 100),
                        "Avg Stops / Run": round(s_sum / mc_runs, 1),
                        "Req. Stops Made/Leg": round(((s_sum / mc_runs) - base_best_stops) / total_legs, 2),
                        "Avg Journey Time": format_time(t_sum / mc_runs, is_duration=True),
                        f"Exp. Consumed ({unit_label})": round(e_sum / mc_runs, 2),
                        f"Savings vs All-Stop ({unit_label})": round(base_worst - e_sum / mc_runs, 2),
                        "Type": "Stochastic",
                    }
                )

            rows_mc.insert(
                0,
                {
                    "Probability": "100% (Worst Case)",
                    "Prob_Num": 100,
                    "Avg Stops / Run": base_worst_stops,
                    "Req. Stops Made/Leg": round(req_per_leg, 2),
                    "Avg Journey Time": format_time(base_worst_time, is_duration=True),
                    f"Exp. Consumed ({unit_label})": round(base_worst, 2),
                    f"Savings vs All-Stop ({unit_label})": 0.0,
                    "Type": "Baseline",
                },
            )
            rows_mc.append(
                {
                    "Probability": "0% (Best Case)",
                    "Prob_Num": 0,
                    "Avg Stops / Run": base_best_stops,
                    "Req. Stops Made/Leg": 0.0,
                    "Avg Journey Time": format_time(base_best_time, is_duration=True),
                    f"Exp. Consumed ({unit_label})": round(base_best, 2),
                    f"Savings vs All-Stop ({unit_label})": round(base_worst - base_best, 2),
                    "Type": "Baseline",
                },
            )

            st.session_state.mc_results = {
                "df": pd.DataFrame(rows_mc),
                "start": mc_start_label,
                "end": mc_end_label,
                "runs": int(mc_runs),
                "unit": unit_label,
                "total_legs": total_legs,
                "cycles": num_cycles,
                "pattern": trip_pattern,
                "mand_per_leg": round(mand_per_leg, 1),
                "req_per_leg": round(req_per_leg, 1),
            }
        except Exception as exc:
            st.error(f"Monte Carlo error: {exc}")

# ============================================================
#  REPRESENTATIVE RUN EXECUTION
# ============================================================
if run_plot:
    if mc_start_label == mc_end_label:
        st.sidebar.error("Start and End must differ!")
        st.stop()

    with st.spinner("Generating representative run telemetry…"):
        try:
            is_primary = plot_dir.startswith(mc_start_label)
            a_label = mc_start_label if is_primary else mc_end_label
            b_label = mc_end_label if is_primary else mc_start_label

            df_profile, s_km, e_km = resolve_profile(a_label, b_label)
            track = make_track(df_profile, s_km, e_km)
            sim = TrainSimulator(mass, length, power, aux_power, accel, decel, traction, efficiency)

            history, stops_names, stats = sim.run_simulation(
                track, s_km, e_km, "random", plot_prob, mc_dwell, record_history=True
            )

            # Collect electrification info for colour band
            electr_segs = [
                (seg["km_high"], seg["km_low"], seg["electrification"])
                for seg in track.segments
            ]

            st.session_state.rep_results = {
                "history": history,
                "stops_names": stops_names,
                "stats": stats,
                "start_name": a_label,
                "end_name": b_label,
                "prob": plot_prob,
                "track": track,
                "traction": traction,
                "efficiency": efficiency,
                "diesel_density": diesel_density,
                "unit": unit_label,
                "electr_segs": electr_segs,
            }
        except Exception as exc:
            st.error(f"Plot run error: {exc}")

# ============================================================
#  TABS
# ============================================================
tab_mc, tab_plot, tab_journey = st.tabs(
    ["🎲 Monte Carlo Fleet Analysis", "📈 Representative Run", "🗺️ Journey Search"]
)

# ── TAB 1: Monte Carlo ───────────────────────────────────────
with tab_mc:
    if st.session_state.mc_results is None:
        st.info("👈 Configure your shift in the sidebar, then click **Run MC**.")
    else:
        mc = st.session_state.mc_results
        st.subheader(f"Monte Carlo Results: {mc['start']}  ➔  {mc['end']}")
        st.markdown(
            f"**Pattern:** {mc['pattern']} &nbsp;|&nbsp; **Cycles:** {mc['cycles']} "
            f"&nbsp;|&nbsp; **Total Legs:** {mc['total_legs']}  \n"
            f"**Stops per Leg:** {mc['mand_per_leg']} mandatory &nbsp;+&nbsp; {mc['req_per_leg']} request"
        )
        savings_col = f"Savings vs All-Stop ({mc['unit']})"
        consumed_col = f"Exp. Consumed ({mc['unit']})"
        display_df = mc["df"].drop(columns=["Prob_Num", "Type"], errors="ignore")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        fig_bar = px.bar(
            mc["df"],
            x="Prob_Num",
            y=savings_col,
            color=savings_col,
            color_continuous_scale="Blues",
            text=savings_col,
            title=f"Expected Savings vs. Worst-Case ({mc['unit']}) by Request-Stop Probability",
            labels={"Prob_Num": "Request Stop Probability (%)"},
        )
        fig_bar.update_traces(width=10, texttemplate="%{text:.2f}", textposition="outside")
        fig_bar.update_layout(
            showlegend=False,
            xaxis=dict(range=[-10, 110], tickvals=[0, 20, 40, 60, 80, 100]),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Energy vs Time scatter
        fig_et = px.scatter(
            mc["df"],
            x="Avg Journey Time",
            y=consumed_col,
            color="Probability",
            symbol="Type",
            size=[12] * len(mc["df"]),
            title="Energy–Time Trade-off by Stopping Policy",
            hover_data=["Avg Stops / Run"],
        )
        st.plotly_chart(fig_et, use_container_width=True)

# ── TAB 2: Representative Run ────────────────────────────────
with tab_plot:
    if st.session_state.rep_results is None:
        st.info("👈 Configure Section 4 in the sidebar, then click **Plot Run**.")
    else:
        r = st.session_state.rep_results
        h = r["history"]
        stats = r["stats"]
        net_unit = (stats["net_kwh"] / (r["diesel_density"] * r["efficiency"])
                    if r["traction"] == "DIESEL" else stats["net_kwh"])

        st.subheader(f"Representative Run: {r['start_name']}  ➔  {r['end_name']}")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Journey Time", format_time(stats["journey_time_s"], is_duration=True))
        col_b.metric(f"Net Consumed ({r['unit']})", f"{net_unit:.2f}")
        col_c.metric("Request-Stop Probability", f"{int(r['prob'] * 100)}%")

        if r["stops_names"]:
            st.markdown("**Stops made:** " + " · ".join(r["stops_names"]))

        x_key = "cum_dist_km" if plot_x_axis == "Distance (km)" else "time_s"
        x_label = "Distance (km)" if plot_x_axis == "Distance (km)" else "Journey Time (s)"

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Speed Profile", "Cumulative Energy"),
            vertical_spacing=0.12,
        )

        fig.add_trace(
            go.Scatter(x=h[x_key], y=h["v_limit"], name="Speed Limit", line=dict(color="red", dash="dot")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=h[x_key], y=h["v_actual"], name="Actual Speed", line=dict(color="royalblue")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=h[x_key], y=h["energy_kwh"], name="Gross Energy (kWh)", line=dict(color="orange")),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(x=h[x_key], y=h["regen_kwh"], name="Recuperated (kWh)", line=dict(color="green", dash="dash")),
            row=2, col=1,
        )

        fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
        fig.update_yaxes(title_text="Energy (kWh)", row=2, col=1)
        fig.update_xaxes(title_text=x_label, row=2, col=1)
        fig.update_layout(height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        # Electrification band info
        if r.get("electr_segs"):
            electr_types = {e for _, _, e in r["electr_segs"] if e != "NONE"}
            if electr_types:
                st.info(f"⚡ Electrification systems on route: **{', '.join(sorted(electr_types))}**")
            else:
                st.info("🛢️ Route is entirely non-electrified (diesel/no overhead).")

        # Gradient and curvature summary from track profile
        track: TrackProfile = r["track"]
        if track.segments:
            grads = [seg["grad"] * 1000 for seg in track.segments]  # back to permille
            curves = [seg["curve_radius"] for seg in track.segments if seg.get("curve_radius")]
            g_col, c_col = st.columns(2)
            g_col.metric("Max Gradient (‰)", f"{max(grads):.1f} ↑  /  {min(grads):.1f} ↓")
            if curves:
                c_col.metric("Minimum Curve Radius (m)", f"{min(curves):.0f}")
            else:
                c_col.metric("Curve Data", "N/A")

# ── TAB 3: Journey Search ────────────────────────────────────
with tab_journey:
    st.header("🗺️ Timetable Journey Search (Dijkstra)")
    st.markdown(
        "Finds the fastest path through the railML timetable graph, "
        "honouring connection constraints and transfer times."
    )

    if not is_railml:
        st.info("Timetable routing requires a railML file containing `<timetable>` / `<trainPart>` data.")
    else:
        j_col1, j_col2, j_col3 = st.columns(3)
        with j_col1:
            j_start = st.selectbox("Origin", station_names, key="j_start")
        with j_col2:
            j_end = st.selectbox("Destination", station_names, key="j_end")
        with j_col3:
            j_time = st.text_input("Earliest Departure (HH:MM)", value="08:00")

        if st.button("🔍 Find Shortest Path"):
            with st.spinner("Parsing timetable and searching…"):
                t_graph = parse_railml_timetable(selected_file)
                if not t_graph.edges:
                    st.error(
                        "No timetable edges found. This railML file may be infrastructure-only "
                        "(no `<trainPart>` timetables)."
                    )
                else:
                    try:
                        journey = find_journey_dijkstra(t_graph, j_start, j_end, j_time)
                        if journey:
                            st.success(f"Path found — total travel time: **{journey['total_time']}**")
                            st.table(pd.DataFrame(journey["path"]))
                            st.session_state.journey_results = journey
                        else:
                            st.warning("No valid connection found at or after the requested departure.")
                    except ValueError as ve:
                        st.error(str(ve))

        if st.session_state.journey_results:
            with st.expander("Previous result"):
                st.table(pd.DataFrame(st.session_state.journey_results["path"]))

# ── Footer ───────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption(
    "Physics: Davis resistance · gradient · curve speed limits · "
    "electrification checks · recuperative braking  \n"
    "Routing: railML 3.x graph (RailMLTrackProfileBuilder) + timetable Dijkstra"
)
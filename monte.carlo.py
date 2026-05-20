import math
import glob
import pandas as pd
import random
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import xml.etree.ElementTree as ET
import heapq
import re
import itertools

# ==========================================
#  PRE-DEFINED VEHICLE FLEET
# ==========================================
PREDEFINED_VEHICLES = {
    "EDITA": {
        "traction": "DIESEL", "mass": 22000, "length": 15,
        "power": 152, "aux_power": 20, "accel": 0.5, "decel": 0.8,
        "efficiency": 0.30
    },
    "EDITA+Btax": {
        "traction": "DIESEL", "mass": 42000, "length": 30,
        "power": 152, "aux_power": 25, "accel": 0.4, "decel": 0.8,
        "efficiency": 0.30
    },
    "RegioNova (Class 814)": {
        "traction": "DIESEL", "mass": 80000, "length": 44,
        "power": 485, "aux_power": 20, "accel": 0.5, "decel": 0.8,
        "efficiency": 0.32
    },
    "Stadler RS1 (Class 840)": {
        "traction": "DIESEL", "mass": 50000, "length": 25.5,
        "power": 514, "aux_power": 25, "accel": 0.8, "decel": 0.9,
        "efficiency": 0.38
    },
    "CityElefant (Class 471)": {
        "traction": "ELECTRIC", "mass": 155000, "length": 79,
        "power": 2000, "aux_power": 80, "accel": 0.8, "decel": 0.9,
        "efficiency": 0.85
    }
}

# ==========================================
#  STREAMLIT PAGE CONFIG & STATE
# ==========================================
st.set_page_config(page_title="Monte Carlo Fleet Simulator", layout="wide")

if "mc_results" not in st.session_state:
    st.session_state.mc_results = None
if "rep_results" not in st.session_state:
    st.session_state.rep_results = None
if "journey_results" not in st.session_state:
    st.session_state.journey_results = None


# ==========================================
#  TIMETABLE & DIJKSTRA UTILITIES
# ==========================================

def parse_time_to_seconds(t_str):
    if not t_str: return None
    try:
        parts = t_str.split(':')
        h = int(parts[0])
        m = int(parts[1])
        s = int(parts[2]) if len(parts) > 2 else 0
        return h * 3600 + m * 60 + s
    except Exception:
        return None


def parse_duration_to_seconds(d_str):
    if not d_str: return 0
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', d_str)
    if match:
        h = int(match.group(1) or 0)
        m = int(match.group(2) or 0)
        s = int(match.group(3) or 0)
        return h * 3600 + m * 60 + s
    return 0


def format_time(sec, is_duration=False):
    if sec is None: return ""
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if is_duration:
        if h > 0:
            return f"{h:02d}h {m:02d}m {s:02d}s"
        return f"{m:02d}m {s:02d}s"
    else:
        h = h % 24
        return f"{h:02d}:{m:02d}:{s:02d}"


class TimetableGraph:
    def __init__(self):
        self.edges = {}
        self.connections = {}
        self.ocp_id_to_name = {}
        self.ocp_name_to_id = {}

    def add_edge(self, from_ocp, to_ocp, tp_id, dep_sec, arr_sec):
        if from_ocp not in self.edges:
            self.edges[from_ocp] = []
        self.edges[from_ocp].append({
            "to": to_ocp,
            "trainPart": tp_id,
            "dep": dep_sec,
            "arr": arr_sec,
            "duration": arr_sec - dep_sec
        })

    def add_connection(self, ocp, from_tp, to_tp, min_time_sec):
        if (ocp, from_tp) not in self.connections:
            self.connections[(ocp, from_tp)] = []
        self.connections[(ocp, from_tp)].append({
            "to_tp": to_tp,
            "min_time": min_time_sec
        })


@st.cache_data
def parse_railml_timetable(filepath):
    """Parses a railML file explicitly for timetable (trainPart) and routing structures."""
    graph = TimetableGraph()
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Strip namespaces dynamically
        for elem in root.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]

        # 1. Map OCPs (Stations) to ID
        for ocp in root.iter('ocp'):
            ocp_id = ocp.get('id')
            ocp_name = ocp.get('name', ocp_id)
            if ocp_id:
                graph.ocp_id_to_name[ocp_id] = ocp_name
                graph.ocp_name_to_id[ocp_name] = ocp_id

        # 2. Parse trainParts & Connection rules
        for tp in root.iter('trainPart'):
            tp_id = tp.get('id')
            ocp_tts = list(tp.findall('.//ocpTT'))

            for i in range(len(ocp_tts) - 1):
                current_ocp = ocp_tts[i]
                next_ocp = ocp_tts[i + 1]

                ocp_A = current_ocp.get('ocpRef')
                ocp_B = next_ocp.get('ocpRef')

                times_A = current_ocp.find('times')
                times_B = next_ocp.find('times')

                if times_A is not None and times_B is not None:
                    dep_sec = parse_time_to_seconds(times_A.get('departure'))
                    arr_sec = parse_time_to_seconds(times_B.get('arrival'))

                    if dep_sec is not None and arr_sec is not None:
                        # Handle overnight transit for this specific edge
                        if arr_sec < dep_sec:
                            arr_sec += 24 * 3600
                        graph.add_edge(ocp_A, ocp_B, tp_id, dep_sec, arr_sec)

                # Parse connections at current_ocp
                for conn in current_ocp.findall('.//connection'):
                    to_tp = conn.get('trainPartRef') or conn.get('trainRef')
                    min_time = parse_duration_to_seconds(conn.get('minConnTime'))
                    if to_tp:
                        graph.add_connection(ocp_A, tp_id, to_tp, min_time)

        return graph
    except Exception as e:
        st.warning(f"Timetable Graph unavailable or error parsing timetable data: {e}")
        return graph


def find_journey_dijkstra(graph: TimetableGraph, start_ocp: str, end_ocp: str, departure_time: str):
    """
    Executes a Dijkstra Shortest Path search honoring railML connection constraints and time weights.
    """
    # Using itertools.count() to ensure heapq never fails on dictionary comparison ties
    tiebreaker = itertools.count()

    start_time = parse_time_to_seconds(departure_time)
    if start_time is None:
        raise ValueError("Invalid departure time format. Please use HH:MM or HH:MM:SS.")

    start_id = graph.ocp_name_to_id.get(start_ocp, start_ocp)
    end_id = graph.ocp_name_to_id.get(end_ocp, end_ocp)

    if start_id not in graph.edges:
        return None

    # Priority Queue State: (current_time, tiebreaker, current_ocp_id, current_trainPart, path)
    pq = []
    heapq.heappush(pq, (start_time, next(tiebreaker), start_id, None, []))

    # Pruning dictionary based on (station, train_part) state
    visited = {}

    while pq:
        current_time, _, current_ocp, current_tp, path = heapq.heappop(pq)

        # Path Found
        if current_ocp == end_id:
            total_duration = current_time - start_time
            formatted_path = []
            for step in path:
                formatted_path.append({
                    "from": graph.ocp_id_to_name.get(step["from"], step["from"]),
                    "to": graph.ocp_id_to_name.get(step["to"], step["to"]),
                    "trainPart": step["trainPart"],
                    "dep": format_time(step["dep"]),
                    "arr": format_time(step["arr"])
                })
            return {
                "total_time": format_time(total_duration, is_duration=True),
                "path": formatted_path
            }

        state = (current_ocp, current_tp)
        if state in visited and visited[state] <= current_time:
            continue
        visited[state] = current_time

        if current_ocp not in graph.edges:
            continue

        # Explore outgoing edges
        for edge in graph.edges[current_ocp]:
            next_tp = edge["trainPart"]
            edge_dep = edge["dep"]
            edge_arr = edge["arr"]

            # Apply Next-Day Wrap around if train departed earlier today
            while edge_dep < current_time:
                edge_dep += 24 * 3600
                edge_arr += 24 * 3600

            # Handle Transfer Logic & Constraints
            if current_tp is not None and current_tp != next_tp:
                allowed = True
                min_transfer = 0

                # Enforce explicit rules if defined for this train/OCP
                conn_key = (current_ocp, current_tp)
                if conn_key in graph.connections:
                    allowed = False
                    for conn in graph.connections[conn_key]:
                        if conn["to_tp"] == next_tp:
                            allowed = True
                            min_transfer = conn["min_time"]
                            break
                else:
                    # Implicit transfer allowed if no strict rule prohibits it
                    min_transfer = 0

                if not allowed:
                    continue

                # Recalculate Next-Day Wrap if transfer time causes us to miss the connection
                while current_time + min_transfer > edge_dep:
                    edge_dep += 24 * 3600
                    edge_arr += 24 * 3600

            # Queue Next State
            next_path = list(path)
            next_path.append({
                "from": current_ocp,
                "to": edge["to"],
                "trainPart": next_tp,
                "dep": edge_dep,
                "arr": edge_arr
            })

            heapq.heappush(pq, (edge_arr, next(tiebreaker), edge["to"], next_tp, next_path))

    return None


# ==========================================
#  DATA LOADERS & PARSERS
# ==========================================

@st.cache_data
def parse_railml_to_dataframe(filepath, track_id=None):
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        for elem in root.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]

        events = []

        for ocp in root.iter('ocp'):
            name = ocp.get('name', '')
            pos = ocp.get('pos')
            if name and pos:
                events.append(
                    {"Poloha": float(pos), "Dopravní bod": name, "Zastavení": "X", "Rychlost": np.nan, "Sklon": np.nan})

        for speed in root.iter('speedChange'):
            pos = speed.get('pos')
            v = speed.get('v')
            if pos and v:
                events.append(
                    {"Poloha": float(pos), "Dopravní bod": "", "Zastavení": "", "Rychlost": float(v), "Sklon": np.nan})

        for grad in root.iter('gradientChange'):
            pos = grad.get('pos')
            slope = grad.get('slope')
            if pos and slope:
                events.append({"Poloha": float(pos), "Dopravní bod": "", "Zastavení": "", "Rychlost": np.nan,
                               "Sklon": float(slope)})

        # Fallback for railML 3.2 Infrastructure structures
        if not events:
            for lps in root.iter('linearPositioningSystem'):
                name_elem = lps.find('name')
                name = name_elem.get('name') if name_elem is not None else lps.get('id')
                start_meas = lps.get('startMeasure')
                end_meas = lps.get('endMeasure')
                if name and start_meas and end_meas:
                    events.append(
                        {"Poloha": float(start_meas) / 1000.0, "Dopravní bod": name, "Zastavení": "X", "Rychlost": 80.0,
                         "Sklon": 0.0})
                    events.append(
                        {"Poloha": float(end_meas) / 1000.0, "Dopravní bod": name + " (End)", "Zastavení": "X",
                         "Rychlost": 80.0, "Sklon": 0.0})

        if not events:
            return pd.DataFrame()

        df = pd.DataFrame(events)
        df = df.sort_values(by="Poloha").reset_index(drop=True)
        df = df.groupby("Poloha", as_index=False).first()
        df["Rychlost"] = df["Rychlost"].ffill().bfill()
        df["Sklon"] = df["Sklon"].ffill().bfill().fillna(0.0)

        return df

    except Exception as e:
        st.error(f"Failed to parse railML file: {e}")
        return pd.DataFrame()


@st.cache_data
def get_unified_dataframe(filepath, track_id=None):
    if filepath.lower().endswith(('.xml', '.railml')):
        return parse_railml_to_dataframe(filepath, track_id)
    else:
        df = pd.read_excel(filepath)
        df.columns = df.columns.str.strip()
        df["Zastavení"] = df["Zastavení"].fillna("")
        df["Poloha"] = pd.to_numeric(df["Poloha"], errors="coerce")
        df["Rychlost"] = pd.to_numeric(df["Rychlost"], errors="coerce")
        df["Sklon"] = pd.to_numeric(df["Sklon"], errors="coerce")
        df = df.dropna(subset=["Poloha"])
        df["Poloha"] = df["Poloha"].apply(lambda x: x / 1000.0 if x > 1000 else x)
        return df


@st.cache_data
def load_mandatory_stations(filepath, track_id=None):
    df = get_unified_dataframe(filepath, track_id)
    if df.empty:
        return []

    stations = []
    for _, row in df.iterrows():
        if str(row["Zastavení"]).strip().upper() == "X":
            name = str(row.get("Dopravní bod", "")).strip()
            if name and name != "nan":
                stations.append((name, row["Poloha"]))

    return sorted(stations, key=lambda x: x[1], reverse=True)


# ==========================================
#  CORE CLASSES
# ==========================================
class TrackProfile:
    def __init__(self, df_raw: pd.DataFrame, is_forward_direction: bool):
        self.is_forward = is_forward_direction
        self.df_raw = self._clean_dataframe(df_raw)
        self.segments = self._build_segments()
        self.stations = self._extract_stations()
        self.station_dict = {s["name"]: s["km"] for s in self.stations}

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by="Poloha", ascending=False).reset_index(drop=True)
        df["Rychlost"] = df["Rychlost"].ffill().bfill()
        df["Sklon"] = df["Sklon"].ffill().bfill().fillna(0)
        return df

    def _build_segments(self) -> list:
        segments = []
        for i in range(len(self.df_raw) - 1):
            km_high = float(self.df_raw.loc[i, "Poloha"])
            km_low = float(self.df_raw.loc[i + 1, "Poloha"])
            v_limit = float(self.df_raw.loc[i, "Rychlost"]) / 3.6
            grad = float(self.df_raw.loc[i, "Sklon"]) / 1000.0
            if not self.is_forward: grad *= -1
            segments.append({"km_high": km_high, "km_low": km_low, "v_limit": v_limit, "grad": grad})
        return segments

    def _extract_stations(self) -> list:
        stations = []
        for _, row in self.df_raw.iterrows():
            name = str(row.get("Dopravní bod", "")).strip()
            stop_type = str(row["Zastavení"]).strip().upper()
            if name and name != "nan" and stop_type in ("X", "R"):
                stations.append({"name": name, "km": row["Poloha"], "type": stop_type})
        return stations

    def get_effective_limit_and_grad(self, front_km: float, rear_km: float) -> tuple:
        span_min, span_max = min(front_km, rear_km), max(front_km, rear_km)
        limits, current_grad, eps = [], 0.0, 1e-6
        for seg in self.segments:
            if span_min <= seg["km_high"] + eps and span_max >= seg["km_low"] - eps:
                limits.append(seg["v_limit"])
            if seg["km_low"] - eps <= front_km <= seg["km_high"] + eps:
                current_grad = seg["grad"]
        return min(limits) if limits else 0.0, current_grad


class TrainSimulator:
    def __init__(self, mass_kg, length_m, max_power_kw, aux_power_kw, max_accel, max_decel, traction_type, efficiency):
        self.mass_kg = mass_kg
        self.eff_mass = self.mass_kg * 1.08
        self.A, self.B, self.C = 1500, 30, 4
        self.traction_efficiency = efficiency if traction_type == "ELECTRIC" else 0.85
        self.regen_efficiency = 0.75 if traction_type == "ELECTRIC" else 0.0
        self.aux_power_w = aux_power_kw * 1000
        self.max_power_w = max_power_kw * 1000
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.train_length_m = length_m

    def get_resistance(self, v_m_s: float) -> float:
        return self.A + (self.B * v_m_s) + (self.C * v_m_s ** 2)

    def _build_events(self, track, start_km, end_km, stop_mode, stop_prob, direction):
        km_min, km_max = min(start_km, end_km), max(start_km, end_km)
        stops_km, stops_names, events = [], [], []

        for station in track.stations:
            km = station["km"]
            if not (km_min <= km <= km_max): continue
            if abs(km - start_km) < 0.01: continue

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
            boundary_km = seg["km_high"] if direction == -1 else seg["km_low"]
            if km_min - 0.01 <= boundary_km <= km_max + 0.01:
                events.append({"km": boundary_km, "target_v": seg["v_limit"], "type": "limit"})

        return events, stops_km, stops_names

    def run_simulation(self, track, start_km, end_km, stop_mode, stop_prob, dwell_time, record_history=False):
        total_dist_m = abs(start_km - end_km) * 1000
        direction = -1 if start_km > end_km else 1
        events, stops_km, stops_names = self._build_events(track, start_km, end_km, stop_mode, stop_prob, direction)

        dt, current_v, distance_covered, total_energy_j, total_regen_j, journey_time_s = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
        g_accel = 9.8186

        history = {"time_s": [], "km": [], "cum_dist_km": [], "v_actual": [], "v_limit": [], "energy_kwh": [],
                   "regen_kwh": []} if record_history else None

        while distance_covered < total_dist_m:
            current_km = start_km + (distance_covered / 1000.0) * direction
            rear_km = current_km - (self.train_length_m / 1000.0) * direction
            effective_limit_m_s, slope = track.get_effective_limit_and_grad(current_km, rear_km)

            f_gradient = self.mass_kg * g_accel * slope
            effective_decel = max(0.05, self.max_decel + (g_accel * slope))
            max_safe_v = effective_limit_m_s

            for event in events:
                tolerance = 0.05 if event["type"] == "stop" else 1e-4
                is_ahead = (event["km"] <= current_km + tolerance) if direction == -1 else (
                        event["km"] >= current_km - tolerance)
                overshot = (current_km < event["km"]) if direction == -1 else (current_km > event["km"])

                if not is_ahead: continue
                dist_to_event = 0.0 if (event["type"] == "stop" and overshot) else abs(current_km - event["km"]) * 1000

                if event["target_v"] < current_v:
                    safe_v = math.sqrt(max(0.0, event["target_v"] ** 2 + 2 * effective_decel * dist_to_event))
                    max_safe_v = min(max_safe_v, safe_v)

            mech_power, regen_power = 0.0, 0.0

            if record_history:
                track_limit_front_m_s, _ = track.get_effective_limit_and_grad(current_km, current_km)
                history["time_s"].append(journey_time_s)
                history["km"].append(current_km)
                history["cum_dist_km"].append(distance_covered / 1000.0)
                history["v_actual"].append(current_v * 3.6)
                history["v_limit"].append(track_limit_front_m_s * 3.6)
                history["energy_kwh"].append(total_energy_j / 3_600_000.0)
                history["regen_kwh"].append(total_regen_j / 3_600_000.0)

            if current_v > max_safe_v + 1e-4:
                f_resist = self.get_resistance(current_v)
                natural_decel = (f_resist + f_gradient) / self.eff_mass
                required_decel = (current_v - max_safe_v) / dt
                brake_application = min(self.max_decel, required_decel - natural_decel)
                brake_application = max(0.0, brake_application)
                actual_decel = brake_application + natural_decel

                regen_power = (self.eff_mass * brake_application) * current_v * self.regen_efficiency
                current_v = max(0.0, current_v - actual_decel * dt)

            elif current_v < min(effective_limit_m_s, max_safe_v) - 1e-4:
                f_resist = self.get_resistance(current_v)
                total_desired_force = f_resist + (self.eff_mass * self.max_accel) + f_gradient
                actual_force = max(0.0, min(total_desired_force, self.max_power_w / max(current_v, 0.5)))
                current_v = max(0.0, min(current_v + ((actual_force - f_resist - f_gradient) / self.eff_mass) * dt,
                                         effective_limit_m_s, max_safe_v))
                mech_power = max(0.0, actual_force * current_v)
            else:
                f_resist = self.get_resistance(current_v)
                total_desired_force = f_resist + f_gradient
                actual_force = max(0.0, min(total_desired_force, self.max_power_w / max(current_v, 0.5)))
                current_v = max(0.0, current_v + ((actual_force - f_resist - f_gradient) / self.eff_mass) * dt)
                mech_power = max(0.0, actual_force * current_v)

            total_energy_j += ((mech_power / self.traction_efficiency) + self.aux_power_w) * dt
            total_regen_j += regen_power * dt
            distance_covered += current_v * dt
            journey_time_s += dt

            if current_v < 0.5 and any(abs(current_km - s) <= 0.05 for s in stops_km):
                current_v = 0.0

                if record_history:
                    for _ in range(2):
                        track_limit_front_m_s, _ = track.get_effective_limit_and_grad(current_km, current_km)
                        history["time_s"].append(journey_time_s)
                        history["km"].append(current_km)
                        history["cum_dist_km"].append(distance_covered / 1000.0)
                        history["v_actual"].append(0.0)
                        history["v_limit"].append(track_limit_front_m_s * 3.6)
                        history["energy_kwh"].append(total_energy_j / 3_600_000.0)
                        history["regen_kwh"].append(total_regen_j / 3_600_000.0)
                        if _ == 0:
                            total_energy_j += self.aux_power_w * dwell_time
                            journey_time_s += dwell_time
                else:
                    total_energy_j += self.aux_power_w * dwell_time
                    journey_time_s += dwell_time

                events = [e for e in events if not (e["type"] == "stop" and abs(e["km"] - current_km) <= 0.05)]
                stops_km = [s for s in stops_km if abs(s - current_km) > 0.05]

        return history, stops_names, {
            "net_kwh": (total_energy_j - total_regen_j) / 3_600_000.0,
            "journey_time_s": journey_time_s,
        }


# ==========================================
#  MAIN DASHBOARD
# ==========================================
st.title("🎲 Monte Carlo Fleet Simulator")
st.markdown(
    "Dedicated tool for calculating stochastic expected values and routing complex Timetable Journeys."
)

# --- 0. FILE LOADER WIDGET ---
st.sidebar.header("📂 Data Source")
excel_files = [f for f in glob.glob("*.xlsx")]
xml_files = [f for f in glob.glob("*.xml")] + [f for f in glob.glob("*.railml")]
all_files = [f for f in excel_files + xml_files if not f.startswith("~$")]

if not all_files:
    st.sidebar.error("No Excel (.xlsx) or railML (.xml/.railml) files found in the current directory.")
    st.stop()

selected_track_file = st.sidebar.selectbox("Select Track Profile", all_files)

selected_track_id = None
if selected_track_file.lower().endswith(('.xml', '.railml')):
    selected_track_id = st.sidebar.text_input("Enter Line ID / Track ID (Optional)",
                                              help="Filters the massive railML graph to a specific route.")

# Load stations from the dynamically selected file
mandatory_stations = load_mandatory_stations(selected_track_file, selected_track_id)
if not mandatory_stations:
    st.error(f"Could not load stations from {selected_track_file}. Ensure the format is supported.")
    st.stop()

station_names = [s[0] for s in mandatory_stations]
station_dict = {s[0]: s[1] for s in mandatory_stations}

# --- 1. TRAIN PARAMETERS ---
st.sidebar.header("1. Train Parameters")
vehicle_choice = st.sidebar.selectbox("Vehicle Profile", ["Custom"] + list(PREDEFINED_VEHICLES.keys()))

if vehicle_choice == "Custom":
    traction = st.sidebar.selectbox("Traction Type", ["DIESEL", "ELECTRIC"], index=0)
    mass = st.sidebar.number_input("Train Mass (kg)", value=100000, step=10000)
    length = st.sidebar.number_input("Train Length (m)", value=40, step=10)
    power = st.sidebar.number_input("Max Power (kW)", value=600, step=50)
    aux_power = st.sidebar.number_input("Auxiliary Power (kW)", value=40, step=5)
    accel = st.sidebar.slider("Max Acceleration (m/s²)", 0.2, 1.2, 0.6, 0.1)
    decel = st.sidebar.slider("Max Braking (m/s²)", 0.4, 1.5, 0.8, 0.1)
    efficiency = st.sidebar.slider("Efficiency (%)", 15, 95, 35) / 100.0
    diesel_density = st.sidebar.number_input("Diesel Density (kWh/L)", value=10.0, step=0.1)
else:
    preset = PREDEFINED_VEHICLES[vehicle_choice]
    traction, mass, length = preset["traction"], preset["mass"], preset["length"]
    power, aux_power = preset["power"], preset["aux_power"]
    accel, decel, efficiency = preset["accel"], preset["decel"], preset["efficiency"]
    diesel_density = 10.0

# --- 2. SHIFT & TRIP BUILDER ---
st.sidebar.header("2. Shift Configuration")
mc_start = st.sidebar.selectbox("Terminal A", station_names, index=0)
mc_end = st.sidebar.selectbox("Terminal B", station_names, index=len(station_names) - 1)

trip_pattern = st.sidebar.radio("Trip Pattern", ["Single Direction (A ➔ B)", "Round Trip (A ➔ B ➔ A)"])
num_cycles = st.sidebar.number_input("Number of Cycles", min_value=1, max_value=20, value=1)

st.sidebar.header("3. Stochastic Settings")
mc_runs = st.sidebar.number_input("N (Runs per Probability)", min_value=10, max_value=1000, value=100, step=10)
mc_dwell = st.sidebar.number_input("Dwell Time (s)", value=30, step=5)

st.sidebar.header("4. Representative Graph")
plot_dir = st.sidebar.radio("Plot Direction", [f"{mc_start} ➔ {mc_end}", f"{mc_end} ➔ {mc_start}"])
plot_prob = st.sidebar.slider("Stop Probability for Plot", 0.0, 1.0, 0.4, 0.1)
plot_x_axis = st.sidebar.radio("Plot X-Axis", ["Distance (km)", "Time (MM:SS)"])

# ==========================================
#  MAIN RUN EXECUTION LOGIC
# ==========================================
c_mc, c_plot = st.sidebar.columns(2)
if c_mc.button("🎲 Run MC", type="primary", use_container_width=True):
    if mc_start == mc_end:
        st.sidebar.error("Start and End stations cannot be the same!")
        st.stop()

    with st.spinner(f"Running Monte Carlo (N={mc_runs}) using {selected_track_file}..."):
        try:
            base_df = get_unified_dataframe(selected_track_file, selected_track_id)
            is_fwd_1 = station_dict[mc_start] > station_dict[mc_end]
            track_1 = TrackProfile(base_df, is_forward_direction=is_fwd_1)
            is_fwd_2 = station_dict[mc_end] > station_dict[mc_start]
            track_2 = TrackProfile(base_df, is_forward_direction=is_fwd_2)

            sim = TrainSimulator(mass, length, power, aux_power, accel, decel, traction, efficiency)


            def get_unit(stats):
                return stats["net_kwh"] / (diesel_density * efficiency) if traction == "DIESEL" else stats["net_kwh"]


            n_fwd = num_cycles
            n_rev = num_cycles if "Round" in trip_pattern else 0
            total_legs = n_fwd + n_rev

            # Baselines
            _, w_stops_1, w_stats_1 = sim.run_simulation(track_1, station_dict[mc_start], station_dict[mc_end], "all",
                                                         1.0, mc_dwell)
            _, b_stops_1, b_stats_1 = sim.run_simulation(track_1, station_dict[mc_start], station_dict[mc_end], "none",
                                                         0.0, mc_dwell)

            if n_rev > 0:
                _, w_stops_2, w_stats_2 = sim.run_simulation(track_2, station_dict[mc_end], station_dict[mc_start],
                                                             "all", 1.0, mc_dwell)
                _, b_stops_2, b_stats_2 = sim.run_simulation(track_2, station_dict[mc_end], station_dict[mc_start],
                                                             "none", 0.0, mc_dwell)
            else:
                w_stops_2, w_stats_2 = [], {"journey_time_s": 0, "net_kwh": 0}
                b_stops_2, b_stats_2 = [], {"journey_time_s": 0, "net_kwh": 0}

            base_worst_unit = (get_unit(w_stats_1) * n_fwd) + (get_unit(w_stats_2) * n_rev)
            base_best_unit = (get_unit(b_stats_1) * n_fwd) + (get_unit(b_stats_2) * n_rev)
            base_worst_time = (w_stats_1["journey_time_s"] * n_fwd) + (w_stats_2["journey_time_s"] * n_rev)
            base_best_time = (b_stats_1["journey_time_s"] * n_fwd) + (b_stats_2["journey_time_s"] * n_rev)
            base_worst_stops = (len(w_stops_1) * n_fwd) + (len(w_stops_2) * n_rev)
            base_best_stops = (len(b_stops_1) * n_fwd) + (len(b_stops_2) * n_rev)

            mand_per_leg = base_best_stops / total_legs
            req_per_leg = (base_worst_stops - base_best_stops) / total_legs

            # Loops
            results = []
            for p in [0.8, 0.6, 0.4, 0.2]:
                t_sum, e_sum, s_sum = 0.0, 0.0, 0.0
                for _ in range(int(mc_runs)):
                    for _f in range(n_fwd):
                        _, r_stops, r_stats = sim.run_simulation(track_1, station_dict[mc_start], station_dict[mc_end],
                                                                 "random", p, mc_dwell)
                        t_sum += r_stats["journey_time_s"];
                        e_sum += get_unit(r_stats);
                        s_sum += len(r_stops)
                    for _r in range(n_rev):
                        _, r_stops, r_stats = sim.run_simulation(track_2, station_dict[mc_end], station_dict[mc_start],
                                                                 "random", p, mc_dwell)
                        t_sum += r_stats["journey_time_s"];
                        e_sum += get_unit(r_stats);
                        s_sum += len(r_stops)

                results.append({
                    "Probability": f"{int(p * 100)}%",
                    "Prob_Num": int(p * 100),
                    "Avg Stops per Run": round(s_sum / mc_runs, 1),
                    "Req. Stops Made/Leg": round(((s_sum / mc_runs) - base_best_stops) / total_legs, 2),
                    "Avg Time": format_time(t_sum / mc_runs, is_duration=True),
                    "Expected Consumed": round(e_sum / mc_runs, 2),
                    "Expected Savings": round(base_worst_unit - (e_sum / mc_runs), 2),
                    "Type": "Stochastic"
                })

            results.insert(0, {
                "Probability": "100% (Worst Case)", "Prob_Num": 100,
                "Avg Stops per Run": base_worst_stops, "Req. Stops Made/Leg": round(req_per_leg, 2),
                "Avg Time": format_time(base_worst_time, is_duration=True),
                "Expected Consumed": round(base_worst_unit, 2), "Expected Savings": 0.0, "Type": "Baseline"
            })
            results.append({
                "Probability": "0% (Best Case)", "Prob_Num": 0,
                "Avg Stops per Run": base_best_stops, "Req. Stops Made/Leg": 0.0,
                "Avg Time": format_time(base_best_time, is_duration=True),
                "Expected Consumed": round(base_best_unit, 2),
                "Expected Savings": round(base_worst_unit - base_best_unit, 2),
                "Type": "Baseline"
            })

            st.session_state.mc_results = {
                "df": pd.DataFrame(results), "start": mc_start, "end": mc_end, "runs": mc_runs,
                "unit": "Liters" if traction == "DIESEL" else "kWh", "total_legs": total_legs,
                "cycles": num_cycles, "pattern": trip_pattern, "mand_per_leg": round(mand_per_leg, 1),
                "req_per_leg": round(req_per_leg, 1)
            }
        except Exception as e:
            st.error(f"Error during Monte Carlo: {e}")

if c_plot.button("📈 Plot Run", use_container_width=True):
    if mc_start == mc_end:
        st.sidebar.error("Start and End stations cannot be the same!")
        st.stop()

    with st.spinner("Generating Representative Run telemetry..."):
        try:
            is_primary_dir = plot_dir.startswith(mc_start)
            p_start = station_dict[mc_start] if is_primary_dir else station_dict[mc_end]
            p_end = station_dict[mc_end] if is_primary_dir else station_dict[mc_start]

            base_df = get_unified_dataframe(selected_track_file, selected_track_id)
            track = TrackProfile(base_df, is_forward_direction=(p_start > p_end))
            sim = TrainSimulator(mass, length, power, aux_power, accel, decel, traction, efficiency)

            history, stops_names, stats = sim.run_simulation(track, p_start, p_end, "random", plot_prob, mc_dwell,
                                                             record_history=True)

            st.session_state.rep_results = {
                "history": history, "stops_names": stops_names, "stats": stats,
                "start_name": mc_start if is_primary_dir else mc_end,
                "end_name": mc_end if is_primary_dir else mc_start,
                "prob": plot_prob, "track": track, "traction": traction,
                "efficiency": efficiency, "diesel_density": diesel_density
            }
        except Exception as e:
            st.error(f"Error generating plot: {e}")

# ==========================================
#  MAIN VIEW RENDERER (TABS)
# ==========================================
tab_mc, tab_plot, tab_journey = st.tabs(
    ["🎲 Monte Carlo Fleet Analysis", "📈 Representative Run Visualization", "🗺️ Journey Search"])

with tab_mc:
    if st.session_state.mc_results is None:
        st.info("👈 **Configure your Shift in the sidebar, then click 'Run MC'.**")
    else:
        mc = st.session_state.mc_results
        st.subheader(f"Monte Carlo Expected Values: {mc['start']} ➔ {mc['end']}")
        st.markdown(
            f"**Configuration:** {mc['pattern']} | **Cycles:** {mc['cycles']} | **Total Legs:** {mc['total_legs']}  \n"
            f"**Stops Per Leg:** {mc['mand_per_leg']} Mandatory | {mc['req_per_leg']} Request")

        display_df = mc["df"].drop(columns=["Prob_Num"])
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        fig = px.bar(mc["df"], x="Prob_Num", y="Expected Savings",
                     title=f"Expected Savings vs. Stopping Policy ({mc['unit']})",
                     labels={"Expected Savings": f"Savings vs. All Stops ({mc['unit']})",
                             "Prob_Num": "Request Stop Probability (%)"},
                     color="Expected Savings", color_continuous_scale="Blues", text="Expected Savings")
        fig.update_traces(width=10, texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(showlegend=False, xaxis=dict(range=[-10, 110], tickvals=[0, 20, 40, 60, 80, 100]),
                          yaxis_range=[0, mc["df"]["Expected Savings"].max() * 1.15] if mc["df"][
                                                                                            "Expected Savings"].max() > 0 else [
                              0, 1])
        st.plotly_chart(fig, use_container_width=True)

with tab_plot:
    if st.session_state.rep_results is None:
        st.info("👈 **Configure parameters in Section 4 of the sidebar, then click 'Plot Run'.**")
    else:
        # Plotly logic (simplified for space)
        pass

with tab_journey:
    st.header("Timetable Journey Search (Dijkstra)")
    st.markdown("Search across the railML timetable graph enforcing train connections and shortest path constraints.")

    if selected_track_file.lower().endswith(('.xml', '.railml')):
        col1, col2, col3 = st.columns(3)
        with col1:
            j_start = st.selectbox("Origin OCP", station_names, key="j_start")
        with col2:
            j_end = st.selectbox("Destination OCP", station_names, key="j_end")
        with col3:
            j_time = st.text_input("Departure Time (HH:MM)", value="08:00")

        if st.button("🔍 Find Shortest Path"):
            with st.spinner("Parsing RailML Timetable constraints and searching..."):
                t_graph = parse_railml_timetable(selected_track_file)
                if not t_graph.edges:
                    st.error(
                        "No valid Timetable routing edges found in this railML file. (Note: RailML 3.x infrastructure-only files do not contain <trainPart> timetables.)")
                else:
                    journey = find_journey_dijkstra(t_graph, j_start, j_end, j_time)
                    if journey:
                        st.success(f"**Path Found!** Total Travel Time: **{journey['total_time']}**")
                        st.table(pd.DataFrame(journey['path']))
                    else:
                        st.warning("No valid path connecting these stations at or after this time was found.")
    else:
        st.info("Timetable Routing requires a valid railML (.xml/.railml) file containing `<timetable>` logic.")
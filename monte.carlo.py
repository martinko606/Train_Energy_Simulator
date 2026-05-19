import math
import glob
import pandas as pd
import random
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import zipfile
import os
import tempfile
from lxml import etree

# ==========================================
#  CONFIGURATION
# ==========================================
# 📍 SPECIFY YOUR ZIP FILE NAME HERE
RAILML_ZIP_PATH = "railML_export_20251214_20260414.zip"

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


# ==========================================
#  RAILML DATA LOADER (FROM ZIP)
# ==========================================
@st.cache_data
def load_railml_from_zip(zip_path):
    """
    Extracts railML XML from a ZIP archive, parses necessary topology using
    memory-efficient iterparse, and builds stations and segments dataframes.
    Filters out non-passenger infrastructure (blocks, junctions).
    """
    if not os.path.exists(zip_path):
        st.error(f"Cannot find ZIP file at: {zip_path}")
        return None, None

    temp_dir = tempfile.mkdtemp()
    xml_file_path = None

    try:
        # 1. Extract ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith('.xml') or file.lower().endswith('.railml'):
                        xml_file_path = os.path.join(root, file)
                        break
                if xml_file_path:
                    break

        if not xml_file_path:
            st.error("No valid .xml or .railml file found inside the ZIP.")
            return None, None

        # 2. Parse XML efficiently using lxml iterparse
        stations = []
        segments = []
        ns = "{*}"

        context = etree.iterparse(xml_file_path, events=('end',), tag=[f"{ns}operationalPoint", f"{ns}line"])

        for event, elem in context:
            tag_name = etree.QName(elem.tag).localname

            # --- Extract Operational Points (Stations & Stops) ---
            if tag_name == 'operationalPoint':
                op_type = None  # Default to None to filter out infrastructure points

                # Check operational type based on DYPOD specs
                op_ops = elem.find(f"{ns}opOperations")
                if op_ops is not None:
                    op_op = op_ops.find(f"{ns}opOperation")
                    if op_op is not None:
                        op_cat = op_op.get("operationalType")

                        if op_cat == "station":
                            op_type = "X"  # Mandatory train station
                        elif op_cat == "stoppingPoint":
                            op_type = "R"  # Request stop (e.g., "zastávka" / "z")

                # If it is a block, junction, crossover, siding, etc., completely ignore it
                if op_type is None:
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
                    continue

                # If it passed the filter, get name and location
                op_name = elem.find(f"{ns}name")
                name_val = op_name.get("name") if op_name is not None else "Unknown"

                spot_loc = elem.find(f"{ns}spotLocation")
                km_val = None
                if spot_loc is not None:
                    lin_coord = spot_loc.find(f"{ns}linearCoordinate")
                    if lin_coord is not None:
                        try:
                            km_val = float(lin_coord.get("measure"))
                        except (ValueError, TypeError):
                            pass

                if km_val is not None:
                    stations.append({
                        "Dopravní bod": name_val,
                        "Poloha": km_val,
                        "Zastavení": op_type
                    })

            # --- Extract Lines (Segments) ---
            elif tag_name == 'line':
                lin_loc = elem.find(f"{ns}linearLocation")
                km_start, km_end = 0.0, 0.0
                if lin_loc is not None:
                    assoc = lin_loc.find(f"{ns}associatedNetElement")
                    if assoc is not None:
                        c_begin = assoc.find(f"{ns}linearCoordinateBegin")
                        c_end = assoc.find(f"{ns}linearCoordinateEnd")
                        if c_begin is not None:
                            try:
                                km_start = float(c_begin.get("measure"))
                            except:
                                pass
                        if c_end is not None:
                            try:
                                km_end = float(c_end.get("measure"))
                            except:
                                pass

                v_limit = 80.0
                perf = elem.find(f"{ns}linePerformance")
                if perf is not None and perf.get("maxSpeed"):
                    try:
                        v_limit = float(perf.get("maxSpeed"))
                    except:
                        pass

                grad = 0.0
                layout = elem.find(f"{ns}lineLayout")
                if layout is not None and layout.get("maxGradient"):
                    try:
                        grad = float(layout.get("maxGradient"))
                    except:
                        pass

                if abs(km_end - km_start) > 0:
                    segments.append({
                        "Poloha_Start": km_start,
                        "Poloha_End": km_end,
                        "Rychlost": v_limit,
                        "Sklon": grad
                    })

            # Free memory
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

        df_stations = pd.DataFrame(stations).drop_duplicates(subset=["Dopravní bod"]).sort_values(by="Poloha",
                                                                                                  ascending=False)
        df_segments = pd.DataFrame(segments)

        return df_stations, df_segments

    finally:
        # 3. Clean up temp directory
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(temp_dir)


# ==========================================
#  CORE CLASSES (Adapted for internal DataFrame)
# ==========================================
class TrackProfile:
    def __init__(self, df_stations: pd.DataFrame, df_segments: pd.DataFrame, is_forward_direction: bool):
        self.is_forward = is_forward_direction
        self.segments = self._build_segments(df_segments)
        self.stations = self._extract_stations(df_stations)
        self.station_dict = {s["name"]: s["km"] for s in self.stations}

    def _build_segments(self, df_segments: pd.DataFrame) -> list:
        segments = []
        for _, row in df_segments.iterrows():
            km_high = max(row["Poloha_Start"], row["Poloha_End"])
            km_low = min(row["Poloha_Start"], row["Poloha_End"])
            v_limit = float(row["Rychlost"]) / 3.6
            grad = float(row["Sklon"]) / 1000.0
            if not self.is_forward: grad *= -1
            segments.append({"km_high": km_high, "km_low": km_low, "v_limit": v_limit, "grad": grad})
        return segments

    def _extract_stations(self, df_stations: pd.DataFrame) -> list:
        stations = []
        for _, row in df_stations.iterrows():
            name = str(row.get("Dopravní bod", "")).strip()
            stop_type = str(row["Zastavení"]).strip().upper()
            if name and name != "nan":
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

        # Build and sort events by distance from start
        events, stops_km, stops_names = self._build_events(track, start_km, end_km, stop_mode, stop_prob, direction)

        # Sort events by how soon we will hit them
        if direction == 1:
            events.sort(key=lambda x: x["km"])
        else:
            events.sort(key=lambda x: x["km"], reverse=True)

        dt = 1.0 if record_history else 2.0  # Double the timestep for background MC runs
        current_v, distance_covered, total_energy_j, total_regen_j, journey_time_s = 0.0, 0.0, 0.0, 0.0, 0.0
        g_accel = 9.8186

        history = {"time_s": [], "km": [], "cum_dist_km": [], "v_actual": [], "v_limit": [], "energy_kwh": [],
                   "regen_kwh": []} if record_history else None

        # Keep track of which event we are looking at to avoid looping through the whole list
        current_event_idx = 0
        total_events = len(events)

        while distance_covered < total_dist_m:
            current_km = start_km + (distance_covered / 1000.0) * direction
            rear_km = current_km - (self.train_length_m / 1000.0) * direction

            # Lookup track properties (this is still slightly slow, but necessary for length)
            effective_limit_m_s, slope = track.get_effective_limit_and_grad(current_km, rear_km)

            f_gradient = self.mass_kg * g_accel * slope
            effective_decel = max(0.05, self.max_decel + (g_accel * slope))
            max_safe_v = effective_limit_m_s

            # FAST EVENT CHECKING: Only look at upcoming events, not past ones
            while current_event_idx < total_events:
                next_event = events[current_event_idx]
                tolerance = 0.05 if next_event["type"] == "stop" else 1e-4

                # Have we passed this event?
                is_ahead = (next_event["km"] <= current_km + tolerance) if direction == -1 else (
                            next_event["km"] >= current_km - tolerance)

                if not is_ahead:
                    current_event_idx += 1  # We passed it, never look at it again
                    continue
                else:
                    break  # We found the immediate next event

            # Look ahead logic (only check the next few upcoming events, not all of them)
            lookahead_limit = min(current_event_idx + 3, total_events)
            for i in range(current_event_idx, lookahead_limit):
                event = events[i]
                overshot = (current_km < event["km"]) if direction == -1 else (current_km > event["km"])
                dist_to_event = 0.0 if (event["type"] == "stop" and overshot) else abs(current_km - event["km"]) * 1000

                if event["target_v"] < current_v:
                    # Braking curve math
                    safe_v = math.sqrt(max(0.0, event["target_v"] ** 2 + 2 * effective_decel * dist_to_event))
                    max_safe_v = min(max_safe_v, safe_v)

            mech_power, regen_power = 0.0, 0.0

            # Record telemetry
            if record_history:
                track_limit_front_m_s, _ = track.get_effective_limit_and_grad(current_km, current_km)
                history["time_s"].append(journey_time_s)
                history["km"].append(current_km)
                history["cum_dist_km"].append(distance_covered / 1000.0)
                history["v_actual"].append(current_v * 3.6)
                history["v_limit"].append(track_limit_front_m_s * 3.6)
                history["energy_kwh"].append(total_energy_j / 3_600_000.0)
                history["regen_kwh"].append(total_regen_j / 3_600_000.0)

            # --- PHYSICS ---
            if current_v > max_safe_v + 1e-4:
                # Braking
                f_resist = self.get_resistance(current_v)
                natural_decel = (f_resist + f_gradient) / self.eff_mass
                required_decel = (current_v - max_safe_v) / dt
                brake_application = min(self.max_decel, required_decel - natural_decel)
                brake_application = max(0.0, brake_application)
                actual_decel = brake_application + natural_decel

                regen_power = (self.eff_mass * brake_application) * current_v * self.regen_efficiency
                current_v = max(0.0, current_v - actual_decel * dt)

            elif current_v < min(effective_limit_m_s, max_safe_v) - 1e-4:
                # Accelerating
                f_resist = self.get_resistance(current_v)
                total_desired_force = f_resist + (self.eff_mass * self.max_accel) + f_gradient

                # Protect against divide by zero at v=0
                actual_force = max(0.0, min(total_desired_force, self.max_power_w / max(current_v, 0.5)))

                current_v = max(0.0, min(current_v + ((actual_force - f_resist - f_gradient) / self.eff_mass) * dt,
                                         effective_limit_m_s, max_safe_v))
                mech_power = max(0.0, actual_force * current_v)
            else:
                # Cruising
                f_resist = self.get_resistance(current_v)
                total_desired_force = f_resist + f_gradient
                actual_force = max(0.0, min(total_desired_force, self.max_power_w / max(current_v, 0.5)))
                current_v = max(0.0, current_v + ((actual_force - f_resist - f_gradient) / self.eff_mass) * dt)
                mech_power = max(0.0, actual_force * current_v)

            total_energy_j += ((mech_power / self.traction_efficiency) + self.aux_power_w) * dt
            total_regen_j += regen_power * dt
            distance_covered += current_v * dt
            journey_time_s += dt

            # Stop handling
            if current_v < 0.5 and current_event_idx < total_events:
                # Did we just hit a stop event?
                close_events = [e for e in events[current_event_idx:current_event_idx + 2] if
                                e["type"] == "stop" and abs(current_km - e["km"]) <= 0.05]

                if close_events:
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

                    # Force the event index forward so we don't get stuck stopping
                    current_event_idx += len(close_events)

        return history, stops_names, {
            "net_kwh": (total_energy_j - total_regen_j) / 3_600_000.0,
            "journey_time_s": journey_time_s,
        }


def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    h = int(m // 60)
    if h > 0:
        return f"{h:02d}h {m % 60:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def create_plotly_figure(history, stops_names, all_stations, is_electric, x_axis_mode):
    # [Unchanged Plotly rendering logic]
    base_time = pd.to_datetime("1970-01-01")
    time_dt_arr = [base_time + pd.to_timedelta(s, unit='s') for s in history["time_s"]]
    dist_arr = history["cum_dist_km"]
    stops_set = set(stops_names)
    net = [g - r for g, r in zip(history["energy_kwh"], history["regen_kwh"])]

    if x_axis_mode == "Time (MM:SS)":
        x_data = time_dt_arr
        total_duration = history["time_s"][-1]
        buffer_seconds = max(30, int(total_duration * 0.03))
        x_min = time_dt_arr[0] - pd.Timedelta(seconds=buffer_seconds)
        x_max = time_dt_arr[-1] + pd.Timedelta(seconds=buffer_seconds)
        x_format = "%H:%M:%S" if total_duration >= 3600 else "%M:%S"
        x_title = "Cumulative Time (HH:MM:SS)" if total_duration >= 3600 else "Cumulative Time (MM:SS)"
    else:
        x_data = dist_arr
        buffer_km = max(0.5, (dist_arr[-1] - dist_arr[0]) * 0.03)
        x_min = dist_arr[0] - buffer_km
        x_max = dist_arr[-1] + buffer_km
        x_title = "Cumulative Distance (km)"
        x_format = None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.25)
    fig.add_trace(go.Scatter(x=x_data, y=history["v_limit"], name="Speed Limit",
                             line=dict(color="#d62728", width=2, dash="dash")), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=x_data, y=history["v_actual"], name="Actual Speed", line=dict(color="#1f77b4", width=2.5),
                   fill="tozeroy", fillcolor="rgba(31, 119, 180, 0.1)"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=x_data, y=history["energy_kwh"], name="Gross Energy", line=dict(color="#ff7f0e", width=2.5)),
        row=2, col=1)
    fig.add_trace(go.Scatter(x=x_data, y=net, name="Net Energy", line=dict(color="#2ca02c", width=2, dash="dot")),
                  row=2, col=1)

    if is_electric:
        fig.add_trace(go.Scatter(x=x_data, y=history["regen_kwh"], name="Regen Recovered",
                                 line=dict(color="#9467bd", width=2, dash="dashdot")), row=2, col=1)

    shapes, annotations = [], []
    v_max = max(max(history["v_limit"]), max(history["v_actual"])) * 1.05
    e_max = max(history["energy_kwh"]) * 1.05

    for station in all_stations:
        if station["type"] not in ["X", "R"]: continue
        skm = station["km"]
        route_min, route_max = min(history["km"]), max(history["km"])
        if not (route_min - 0.05 <= skm <= route_max + 0.05): continue
        color = "gray" if station["type"] == "X" else ("black" if station["name"] in stops_set else "#d62728")

        try:
            idx = (np.abs(np.array(history["km"]) - skm)).argmin()
            x_pos = time_dt_arr[idx] if x_axis_mode == "Time (MM:SS)" else dist_arr[idx]

            shapes.append(dict(type="line", xref="x", yref="y", x0=x_pos, x1=x_pos, y0=0, y1=v_max,
                               line=dict(color=color, width=1, dash="dot"), layer="below"))
            shapes.append(dict(type="line", xref="x", yref="y2", x0=x_pos, x1=x_pos, y0=0, y1=e_max,
                               line=dict(color=color, width=1, dash="dot"), layer="below"))

            annotations.append(
                dict(x=x_pos, y=-0.08, xref="x", yref="y domain", text=station["name"].title(), showarrow=False,
                     font=dict(family="Times New Roman, serif", size=12, color=color), xanchor="right", yanchor="top",
                     textangle=-45))
            annotations.append(
                dict(x=x_pos, y=-0.08, xref="x", yref="y2 domain", text=station["name"].title(), showarrow=False,
                     font=dict(family="Times New Roman, serif", size=12, color=color), xanchor="right", yanchor="top",
                     textangle=-45))
        except ValueError:
            pass

    fig.update_layout(template="simple_white", font=dict(family="Times New Roman, serif", size=14, color="black"),
                      height=900, margin=dict(l=60, r=40, t=40, b=180), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      xaxis=dict(showgrid=True, gridcolor='lightgray', title=dict(text=x_title, standoff=80),
                                 range=[x_min, x_max], tickformat=x_format, showticklabels=True, linecolor='black',
                                 mirror=True),
                      xaxis2=dict(showgrid=True, gridcolor='lightgray', title=dict(text=x_title, standoff=80),
                                  range=[x_min, x_max], tickformat=x_format, showticklabels=True, linecolor='black',
                                  mirror=True),
                      yaxis=dict(title="Speed (km/h)", showgrid=True, gridcolor='lightgray', linecolor='black',
                                 mirror=True),
                      yaxis2=dict(title="Energy (kWh)", showgrid=True, gridcolor='lightgray', linecolor='black',
                                  mirror=True), shapes=shapes, annotations=annotations)
    return fig


# ==========================================
#  MAIN DASHBOARD
# ==========================================
st.title("🎲 Monte Carlo Fleet Simulator")
st.markdown(
    "Dedicated tool for calculating stochastic expected values and rendering representative kinematic profiles.")

# --- 0. DATA LOAD ---
with st.spinner("Loading railML data from ZIP..."):
    df_stations, df_segments = load_railml_from_zip(RAILML_ZIP_PATH)

if df_stations is None or df_stations.empty:
    st.error("Failed to load or parse railML data. Please ensure the ZIP file exists and contains valid railML.")
    st.stop()

# Build dropdown lists
mandatory_stations = df_stations[df_stations["Zastavení"] == "X"][["Dopravní bod", "Poloha"]].values.tolist()
station_names = df_stations["Dopravní bod"].tolist()
station_dict = dict(zip(df_stations["Dopravní bod"], df_stations["Poloha"]))

st.sidebar.header("📂 Data Source")
st.sidebar.success(f"Loaded network from {RAILML_ZIP_PATH}")

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

    if traction == "DIESEL":
        efficiency = st.sidebar.slider("Thermal Efficiency (%)", 15, 60, 35) / 100.0
        diesel_density = st.sidebar.number_input("Diesel Energy Density (kWh/L)", value=10.0, step=0.1)
    else:
        efficiency = st.sidebar.slider("Grid-to-Wheel Efficiency (%)", 50, 95, 85) / 100.0
        diesel_density = 10.0
else:
    preset = PREDEFINED_VEHICLES[vehicle_choice]
    traction = preset["traction"]
    mass = preset["mass"]
    length = preset["length"]
    power = preset["power"]
    aux_power = preset["aux_power"]
    accel = preset["accel"]
    decel = preset["decel"]
    efficiency = preset["efficiency"]

    diesel_density = st.sidebar.number_input("Diesel Energy Density (kWh/L)", value=10.0,
                                             step=0.1) if traction == "DIESEL" else 10.0

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

# --- RUN BUTTONS ---
c_mc, c_plot = st.sidebar.columns(2)

if c_mc.button("🎲 Run MC", type="primary", use_container_width=True):
    if mc_start == mc_end:
        st.sidebar.error("Start and End stations cannot be the same!")
        st.stop()

    with st.spinner(f"Running Monte Carlo (N={mc_runs})..."):
        try:
            is_fwd_1 = station_dict[mc_start] > station_dict[mc_end]
            track_1 = TrackProfile(df_stations, df_segments, is_forward_direction=is_fwd_1)
            is_fwd_2 = station_dict[mc_end] > station_dict[mc_start]
            track_2 = TrackProfile(df_stations, df_segments, is_forward_direction=is_fwd_2)

            sim = TrainSimulator(mass, length, power, aux_power, accel, decel, traction, efficiency)


            def get_unit(stats):
                return stats["net_kwh"] / (diesel_density * efficiency) if traction == "DIESEL" else stats["net_kwh"]


            n_fwd = num_cycles
            n_rev = num_cycles if trip_pattern == "Round Trip (A ➔ B ➔ A)" else 0
            total_legs = n_fwd + n_rev

            _, w_stops_1, w_stats_1 = sim.run_simulation(track_1, station_dict[mc_start], station_dict[mc_end], "all",
                                                         1.0, mc_dwell, record_history=False)
            _, b_stops_1, b_stats_1 = sim.run_simulation(track_1, station_dict[mc_start], station_dict[mc_end], "none",
                                                         0.0, mc_dwell, record_history=False)

            if n_rev > 0:
                _, w_stops_2, w_stats_2 = sim.run_simulation(track_2, station_dict[mc_end], station_dict[mc_start],
                                                             "all", 1.0, mc_dwell, record_history=False)
                _, b_stops_2, b_stats_2 = sim.run_simulation(track_2, station_dict[mc_end], station_dict[mc_start],
                                                             "none", 0.0, mc_dwell, record_history=False)
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

            results = []
            for p in [0.8, 0.6, 0.4, 0.2]:
                t_sum, e_sum, s_sum = 0.0, 0.0, 0.0
                for _ in range(int(mc_runs)):
                    for _f in range(n_fwd):
                        _, r_stops, r_stats = sim.run_simulation(track_1, station_dict[mc_start], station_dict[mc_end],
                                                                 "random", p, mc_dwell, record_history=False)
                        t_sum += r_stats["journey_time_s"]
                        e_sum += get_unit(r_stats)
                        s_sum += len(r_stops)

                    for _r in range(n_rev):
                        _, r_stops, r_stats = sim.run_simulation(track_2, station_dict[mc_end], station_dict[mc_start],
                                                                 "random", p, mc_dwell, record_history=False)
                        t_sum += r_stats["journey_time_s"]
                        e_sum += get_unit(r_stats)
                        s_sum += len(r_stops)

                results.append({"Probability": f"{int(p * 100)}%", "Prob_Num": int(p * 100),
                                "Avg Stops per Run": round(s_sum / mc_runs, 1),
                                "Req. Stops Made/Leg": round(((s_sum / mc_runs) - base_best_stops) / total_legs, 2),
                                "Avg Time": format_time(t_sum / mc_runs),
                                "Expected Consumed": round(e_sum / mc_runs, 2),
                                "Expected Savings": round(base_worst_unit - (e_sum / mc_runs), 2),
                                "Type": "Stochastic"})

            results.insert(0,
                           {"Probability": "100% (Worst Case)", "Prob_Num": 100, "Avg Stops per Run": base_worst_stops,
                            "Req. Stops Made/Leg": round(req_per_leg, 2), "Avg Time": format_time(base_worst_time),
                            "Expected Consumed": round(base_worst_unit, 2), "Expected Savings": 0.0,
                            "Type": "Baseline"})
            results.append({"Probability": "0% (Best Case)", "Prob_Num": 0, "Avg Stops per Run": base_best_stops,
                            "Req. Stops Made/Leg": 0.0, "Avg Time": format_time(base_best_time),
                            "Expected Consumed": round(base_best_unit, 2),
                            "Expected Savings": round(base_worst_unit - base_best_unit, 2), "Type": "Baseline"})

            st.session_state.mc_results = {"df": pd.DataFrame(results), "start": mc_start, "end": mc_end,
                                           "runs": mc_runs, "unit": "Liters" if traction == "DIESEL" else "kWh",
                                           "total_legs": total_legs, "cycles": num_cycles, "pattern": trip_pattern,
                                           "mand_per_leg": round(mand_per_leg, 1), "req_per_leg": round(req_per_leg, 1)}
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
            is_fwd = p_start > p_end
            track = TrackProfile(df_stations, df_segments, is_forward_direction=is_fwd)
            sim = TrainSimulator(mass, length, power, aux_power, accel, decel, traction, efficiency)

            history, stops_names, stats = sim.run_simulation(track, p_start, p_end, "random", plot_prob, mc_dwell,
                                                             record_history=True)
            st.session_state.rep_results = {"history": history, "stops_names": stops_names, "stats": stats,
                                            "start_name": mc_start if is_primary_dir else mc_end,
                                            "end_name": mc_end if is_primary_dir else mc_start, "prob": plot_prob,
                                            "track": track, "traction": traction, "efficiency": efficiency,
                                            "diesel_density": diesel_density}
        except Exception as e:
            st.error(f"Error generating plot: {e}")

# ==========================================
#  MAIN VIEW RENDERER (TABS)
# ==========================================
tab_mc, tab_plot = st.tabs(["🎲 Monte Carlo Fleet Analysis", "📈 Representative Run Visualization"])

with tab_mc:
    if st.session_state.mc_results is None:
        st.info("👈 **Configure your Shift in the sidebar, then click 'Run MC'.**")
    else:
        mc = st.session_state.mc_results
        df = mc["df"]
        unit = mc["unit"]

        st.subheader(f"Monte Carlo Expected Values: {mc['start']} ➔ {mc['end']}")
        st.markdown(
            f"**Configuration:** {mc['pattern']} | **Cycles:** {mc['cycles']} | **Total Legs:** {mc['total_legs']}  \n**Stops Per Leg:** {mc['mand_per_leg']} Mandatory | {mc['req_per_leg']} Request")
        display_df = df.drop(columns=["Prob_Num"])
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        fig = px.bar(df, x="Prob_Num", y="Expected Savings", title=f"Expected Savings vs. Stopping Policy ({unit})",
                     labels={"Expected Savings": f"Savings vs. All Stops ({unit})",
                             "Prob_Num": "Request Stop Probability (%)"}, color="Expected Savings",
                     color_continuous_scale="Blues", text="Expected Savings")
        fig.update_traces(width=10, texttemplate='%{text:.2f}', textposition='outside')
        max_y = df["Expected Savings"].max()
        fig.update_layout(showlegend=False, xaxis=dict(range=[-10, 110], tickvals=[0, 20, 40, 60, 80, 100]),
                          yaxis_range=[0, max_y * 1.15] if max_y > 0 else [0, 1])
        st.plotly_chart(fig, use_container_width=True)

with tab_plot:
    if st.session_state.rep_results is None:
        st.info("👈 **Configure parameters in Section 4 of the sidebar, then click 'Plot Run'.**")
    else:
        rep = st.session_state.rep_results
        st.subheader(f"Representative Run: {rep['start_name']} ➔ {rep['end_name']} (P = {int(rep['prob'] * 100)}%)")

        m, s = int(rep["stats"]["journey_time_s"] // 60), int(rep["stats"]["journey_time_s"] % 60)
        c1, c2, c3 = st.columns(3)
        c1.metric("Run Time", f"{m:02d}:{s:02d}")
        c2.metric("Stops Made", len(rep["stops_names"]))

        if rep["traction"] == "DIESEL":
            consumed = rep["stats"]["net_kwh"] / (rep["diesel_density"] * rep["efficiency"])
            c3.metric("Fuel Consumed", f"{consumed:.1f} L")
        else:
            c3.metric("Energy Consumed", f"{rep['stats']['net_kwh']:.1f} kWh")

        fig = create_plotly_figure(rep["history"], rep["stops_names"], rep["track"].stations,
                                   rep["traction"] == "ELECTRIC", plot_x_axis)
        st.plotly_chart(fig, use_container_width=True, theme=None)
        st.caption(
            f"**Stops Made during this specific run:** {', '.join(rep['stops_names']) if rep['stops_names'] else 'None'}")
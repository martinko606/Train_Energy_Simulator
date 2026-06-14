import math
import os
import glob
import pandas as pd
import random
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import zipfile
import tempfile
from lxml import etree
import heapq

# ==========================================
#  CONFIGURATION
# ==========================================
RAILML_ZIP_PATH = "railML_export_20251214_20260414.zip"

# ==========================================
#  PRE-DEFINED VEHICLE FLEET
# ==========================================
PREDEFINED_VEHICLES = {
    "EDITA": {
        "traction": "DIESEL", "mass": 22000, "length": 15,
        "power": 250, "aux_power": 20, "accel": 0.5, "decel": 0.8,
        "efficiency": 0.30, "max_speed": 80
    },
    "EDITA+Btax": {
        "traction": "DIESEL", "mass": 42000, "length": 30,
        "power": 250, "aux_power": 25, "accel": 0.4, "decel": 0.8,
        "efficiency": 0.30, "max_speed": 80
    },
    "Regiopanter 3 car (Class 640)": {
        "traction": "ELECTRIC", "mass": 159000, "length": 79.4,
        "power": 2040, "aux_power": 80, "accel": 0.8, "decel": 0.9,
        "efficiency": 0.85, "max_speed": 160
    },
    "Regionova (Class 814)": {
        "traction": "DIESEL", "mass": 39600, "length": 28.44,
        "power": 242, "aux_power": 10, "accel": 0.5, "decel": 0.8,
        "efficiency": 0.30, "max_speed": 80
    },
    "750-7 + 3 cars": {
        "traction": "DIESEL", "mass": 207000, "length": 90.16,
        "power": 1550, "aux_power": 40, "accel": 0.4, "decel": 0.8,
        "efficiency": 0.30, "max_speed": 100
    },
}

# ==========================================
#  STREAMLIT PAGE CONFIG & STATE
# ==========================================
st.set_page_config(page_title="Railway Energy Simulator", layout="wide")

if "journey_results" not in st.session_state:
    st.session_state.journey_results = []
if "mc_results" not in st.session_state:
    st.session_state.mc_results = None


# ==========================================
#  RAILML DATA LOADER (TOPOLOGICAL GRAPH)
# ==========================================
@st.cache_data
def load_railml_from_zip(zip_path):
    """
    Extracts railML XML and builds a mathematical graph of the rail network.
    Returns Nodes (Stations/Junctions) and Edges (Micro-segments).
    """
    if not os.path.exists(zip_path):
        return None, None

    temp_dir = tempfile.mkdtemp()
    xml_file_path = None

    try:
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
            return None, None

        stations_dict = {}
        lines_dict = {}
        ns = "{*}"

        context = etree.iterparse(xml_file_path, events=('end',), tag=[f"{ns}operationalPoint", f"{ns}line"])

        for event, elem in context:
            tag_name = etree.QName(elem.tag).localname

            # --- Extract Operational Points (Nodes) ---
            if tag_name == 'operationalPoint':
                op_type = "IGNORE"

                op_ops = elem.find(f"{ns}opOperations")
                if op_ops is not None:
                    op_op = op_ops.find(f"{ns}opOperation")
                    if op_op is not None:
                        op_cat = op_op.get("operationalType")
                        if op_cat == "station":
                            op_type = "X"  # Mandatory Stop
                        elif op_cat == "stoppingPoint":
                            op_type = "R"  # Request Stop

                op_id = elem.get("id")
                op_name = elem.find(f"{ns}name")
                name_val = op_name.get("name") if op_name is not None else "Unknown"

                connected_lines = []
                for c in elem.findall(f"{ns}connectedToLine"):
                    ref = c.get("ref")
                    if ref: connected_lines.append(ref)

                stations_dict[op_id] = {
                    "name": name_val,
                    "type": op_type,
                    "connected": connected_lines
                }

            # --- Extract Lines (Edges/Micro-Segments) ---
            elif tag_name == 'line':
                line_id = elem.get("id")

                # Extract Line Name based on user preference instead of designator
                line_name = "Unknown"
                line_name_elem = elem.find(f"{ns}name")
                if line_name_elem is not None and line_name_elem.get("name"):
                    line_name = line_name_elem.get("name").strip()

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

                lines_dict[line_id] = {
                    "km_start": km_start,
                    "km_end": km_end,
                    "v_limit": v_limit,
                    "grad": grad,
                    "endpoints": [],
                    "line_name": line_name
                }

            # Free memory dynamically
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

        # --- Resolve Graph Endpoints ---
        for op_id, data in stations_dict.items():
            for ref in data["connected"]:
                if ref in lines_dict:
                    if op_id not in lines_dict[ref]["endpoints"]:
                        lines_dict[ref]["endpoints"].append(op_id)

        valid_edges = []
        for l_id, l_data in lines_dict.items():
            endpoints = l_data.get("endpoints", [])
            # Only process lines that cleanly connect two nodes
            if len(endpoints) == 2:
                length = abs(l_data["km_end"] - l_data["km_start"])
                if length == 0: length = 0.05  # Failsafe for missing length
                valid_edges.append({
                    "u": endpoints[0],
                    "v": endpoints[1],
                    "length": length,
                    "v_limit": l_data["v_limit"],
                    "grad": l_data["grad"],
                    "line_name": l_data["line_name"]
                })

        return stations_dict, valid_edges

    finally:
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files: os.remove(os.path.join(root, name))
            for name in dirs: os.rmdir(os.path.join(root, name))
        os.rmdir(temp_dir)


# ==========================================
#  TOPOLOGICAL PATHFINDER (DIJKSTRA)
# ==========================================
def build_route_profile(start_name, end_name, nodes, edges, is_reverse=False):
    """
    Searches the graph for the shortest route between two stations and stitches
    the micro-segments into a single contiguous 1D Track Profile.
    """
    # 1. Map Names to Graph IDs
    start_id = next((nid for nid, n in nodes.items() if n["name"] == start_name), None)
    end_id = next((nid for nid, n in nodes.items() if n["name"] == end_name), None)

    if not start_id or not end_id:
        return None

    # 2. Build Adjacency Matrix
    adj = {nid: [] for nid in nodes}
    for e in edges:
        adj[e["u"]].append((e["v"], e))
        adj[e["v"]].append((e["u"], e))

    # 3. Dijkstra's Algorithm
    queue = [(0.0, start_id, [], [])]
    visited = set()
    n_path, e_path = None, None

    while queue:
        dist, current, np_path, ep_path = heapq.heappop(queue)
        if current in visited: continue
        visited.add(current)

        new_np = np_path + [current]
        if current == end_id:
            n_path, e_path = new_np, ep_path
            break

        for neighbor, edge in adj.get(current, []):
            if neighbor not in visited:
                heapq.heappush(queue, (dist + edge["length"], neighbor, new_np, ep_path + [edge]))

    if not n_path:
        return None  # No route found

    # 4. Construct Contiguous 1D Profile
    path_stations = []
    path_segments = []
    cum_dist = 0.0

    for i in range(len(n_path)):
        nid = n_path[i]

        # Only add valid passenger stops to the physics array (Ignore junctions)
        if nodes[nid]["type"] in ["X", "R"]:
            path_stations.append({
                "name": nodes[nid]["name"],
                "km": cum_dist,
                "type": nodes[nid]["type"]
            })

        if i < len(n_path) - 1:
            edge = e_path[i]
            path_segments.append({
                "km_low": cum_dist,
                "km_high": cum_dist + edge["length"],
                "v_limit": edge["v_limit"],
                "grad": edge["grad"]
            })
            cum_dist += edge["length"]

    return TrackProfile(path_stations, path_segments, is_reverse=is_reverse)


# ==========================================
#  CORE CLASSES
# ==========================================
class TrackProfile:
    def __init__(self, path_stations: list, path_segments: list, is_reverse: bool):
        self.stations = path_stations
        self.segments = []

        # Assemble track segments (flip gradients if this is the reverse route)
        for seg in path_segments:
            grad = -seg["grad"] if is_reverse else seg["grad"]
            self.segments.append({
                "km_low": seg["km_low"],
                "km_high": seg["km_high"],
                "v_limit": seg["v_limit"],
                "grad": grad
            })
        self.station_dict = {s["name"]: s["km"] for s in self.stations}

    def get_effective_limit_and_grad(self, front_km: float, rear_km: float) -> tuple:
        if not self.segments:
            return 80.0 / 3.6, 0.0

        span_min, span_max = min(front_km, rear_km), max(front_km, rear_km)
        limits, current_grad = [], 0.0
        eps = 1e-6

        # Ensure we don't evaluate out of bounds
        span_min = max(span_min, self.segments[0]["km_low"])
        span_max = min(span_max, self.segments[-1]["km_high"])

        for seg in self.segments:
            if span_min <= seg["km_high"] + eps and span_max >= seg["km_low"] - eps:
                limits.append(seg["v_limit"])

            # Evaluate gradient at center of mass
            center_km = (front_km + rear_km) / 2.0
            center_km = max(self.segments[0]["km_low"], min(center_km, self.segments[-1]["km_high"]))
            if seg["km_low"] - eps <= center_km <= seg["km_high"] + eps:
                current_grad = seg["grad"]

        if limits:
            self.last_limit = min(limits)
        return getattr(self, 'last_limit', 80.0 / 3.6), current_grad


class TrainSimulator:
    def __init__(self, mass_kg, length_m, max_power_kw, aux_power_kw, max_accel, max_decel, traction_type, efficiency,
                 max_speed_kmh):
        self.mass_kg = mass_kg
        self.eff_mass = self.mass_kg * 1.08
        self.traction_efficiency = efficiency if traction_type == "ELECTRIC" else 0.85
        self.regen_efficiency = 0.75 if traction_type == "ELECTRIC" else 0.0
        self.aux_power_w = aux_power_kw * 1000
        self.max_power_w = max_power_kw * 1000
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.train_length_m = length_m
        self.max_speed_m_s = max_speed_kmh / 3.6

        # Dynamically calculate Davis Equation coefficients based on train mass and length
        self.mass_tons = self.mass_kg / 1000.0
        self.A = 20.0 * self.mass_tons  # Static/Mechanical resistance
        self.B = 0.2 * self.mass_tons  # Linear resistance (flange/momentum)
        self.C = 2.5 + (0.03 * self.train_length_m)  # Aerodynamic drag

    def get_resistance(self, v_m_s: float) -> float:
        return self.A + (self.B * v_m_s) + (self.C * v_m_s ** 2)

    def _build_events(self, track, start_km, end_km, stop_mode, stop_prob, direction):
        stops_km, stops_names, events = [], [], []

        for station in track.stations:
            km = station["km"]
            if abs(km - start_km) < 0.01:
                continue  # Don't stop immediately where we started

            will_stop = False
            if station["type"] == "X":
                will_stop = True
            elif station["type"] == "R":
                if stop_mode == "all":
                    will_stop = True
                elif stop_mode == "random" and random.random() <= stop_prob:
                    will_stop = True

            # Force stop at destination terminal
            if abs(km - end_km) < 0.01:
                will_stop = True

            if will_stop:
                stops_km.append(km)
                stops_names.append(station["name"])
                events.append({"km": km, "target_v": 0.0, "type": "stop"})

        for seg in track.segments:
            boundary_km = seg["km_high"] if direction == -1 else seg["km_low"]
            events.append({"km": boundary_km, "target_v": seg["v_limit"], "type": "limit"})

        return events, stops_km, stops_names

    def run_simulation(self, track, start_km, end_km, stop_mode, stop_prob, dwell_time, global_start_time=0.0,
                       global_start_dist=0.0):
        total_dist_m = abs(start_km - end_km) * 1000
        direction = -1 if start_km > end_km else 1

        events, stops_km, stops_names = self._build_events(track, start_km, end_km, stop_mode, stop_prob, direction)
        events.sort(key=lambda x: x["km"], reverse=(direction == -1))

        dt = 1.0
        current_v, distance_covered, total_energy_j, total_regen_j = 0.0, 0.0, 0.0, 0.0

        journey_time_s = global_start_time
        history = {"time_s": [], "km": [], "cum_dist_km": [], "v_actual": [], "v_limit": [], "energy_kwh": [],
                   "regen_kwh": []}

        g_accel = 9.8186
        current_event_idx = 0
        total_events = len(events)
        stall_counter = 0
        MAX_JOURNEY_TIME = 14400

        while distance_covered < total_dist_m:

            # --- Failsafe Monitoring ---
            if journey_time_s > MAX_JOURNEY_TIME:
                st.warning("Simulation aborted: Journey exceeded maximum safe limit of 4 hours.")
                break

            if current_v < 0.05 and current_event_idx >= total_events:
                if (total_dist_m - distance_covered) < 50:
                    distance_covered = total_dist_m
                    break
                stall_counter += dt
                if stall_counter > 120:
                    st.warning("Simulation aborted: Train stalled. Likely insufficient power for the gradient.")
                    break
            else:
                stall_counter = 0

            current_km = start_km + (distance_covered / 1000.0) * direction
            rear_km = current_km - (self.train_length_m / 1000.0) * direction

            # Fetch limits and cap to Train's Maximum Permitted Speed
            effective_limit_m_s, slope = track.get_effective_limit_and_grad(current_km, rear_km)
            effective_limit_m_s = min(effective_limit_m_s, self.max_speed_m_s)

            track_limit_front_m_s, _ = track.get_effective_limit_and_grad(current_km, current_km)
            track_limit_front_m_s = min(track_limit_front_m_s, self.max_speed_m_s)

            f_gradient = self.mass_kg * g_accel * slope
            effective_decel = max(0.05, self.max_decel + (g_accel * slope))
            max_safe_v = effective_limit_m_s

            while current_event_idx < total_events:
                e = events[current_event_idx]
                dist_to_e_km = (e["km"] - current_km) * direction
                if dist_to_e_km < -0.001:
                    current_event_idx += 1
                else:
                    break

            lookahead_limit = min(current_event_idx + 3, total_events)
            for i in range(current_event_idx, lookahead_limit):
                event = events[i]
                dist_to_event = max(0.0, (event["km"] - current_km) * direction) * 1000

                if event["type"] == "stop":
                    dist_to_event = max(0.0, dist_to_event - 2.0)

                if event["target_v"] < current_v:
                    safe_v = math.sqrt(max(0.0, event["target_v"] ** 2 + 2 * effective_decel * dist_to_event))
                    max_safe_v = min(max_safe_v, safe_v)

            mech_power, regen_power = 0.0, 0.0

            history["time_s"].append(journey_time_s)
            history["km"].append(current_km)
            history["cum_dist_km"].append(global_start_dist + (distance_covered / 1000.0))
            history["v_actual"].append(current_v * 3.6)
            history["v_limit"].append(track_limit_front_m_s * 3.6)
            history["energy_kwh"].append(total_energy_j / 3_600_000.0)
            history["regen_kwh"].append(total_regen_j / 3_600_000.0)

            # Physics Engine
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

            # Stop Event Execution
            if current_v < 0.5 and current_event_idx < total_events:
                next_e = events[current_event_idx]
                if next_e["type"] == "stop":
                    dist_to_e_km = (next_e["km"] - current_km) * direction

                    if -0.01 <= dist_to_e_km <= 0.025:
                        current_v = 0.0
                        for _ in range(2):
                            history["time_s"].append(journey_time_s)
                            history["km"].append(current_km)
                            history["cum_dist_km"].append(global_start_dist + (distance_covered / 1000.0))
                            history["v_actual"].append(0.0)
                            history["v_limit"].append(track_limit_front_m_s * 3.6)
                            history["energy_kwh"].append(total_energy_j / 3_600_000.0)
                            history["regen_kwh"].append(total_regen_j / 3_600_000.0)
                            if _ == 0:
                                total_energy_j += self.aux_power_w * dwell_time
                                journey_time_s += dwell_time

                        current_event_idx += 1
                        if abs(current_km - end_km) <= 0.05:
                            distance_covered = total_dist_m

        return history, stops_names, {
            "total_kwh": total_energy_j / 3_600_000.0,
            "regen_kwh": total_regen_j / 3_600_000.0,
            "net_kwh": (total_energy_j - total_regen_j) / 3_600_000.0,
            "journey_time_s": journey_time_s,
        }


def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    h = int(m // 60)
    if h > 0: return f"{h:02d}:{m % 60:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def create_plotly_figure(history, stops_names, all_stations, is_electric, x_axis_mode):
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
                             line=dict(color="#ff4b4b", dash="dash")), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=x_data, y=history["v_actual"], name="Actual Speed", line=dict(color="#0068c9", width=2),
                   fill="tozeroy", fillcolor="rgba(0, 104, 201, 0.1)"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=x_data, y=history["energy_kwh"], name="Gross Energy", line=dict(color="#f9a03f", width=2),
                   fill="tozeroy", fillcolor="rgba(249, 160, 63, 0.1)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x_data, y=net, name="Net Energy", line=dict(color="#83c9ff", width=2, dash="dashdot")),
                  row=2, col=1)

    if is_electric:
        fig.add_trace(go.Scatter(x=x_data, y=history["regen_kwh"], name="Regen Recovered",
                                 line=dict(color="#29b5e8", width=1.5, dash="dot"), fill="tozeroy",
                                 fillcolor="rgba(41, 181, 232, 0.1)"), row=2, col=1)

    shapes, annotations = [], []
    v_max = max(max(history["v_limit"]), max(history["v_actual"])) * 1.05
    e_max = max(history["energy_kwh"]) * 1.05

    for station in all_stations:
        if station["type"] not in ["X", "R"]: continue
        skm = station["km"]
        route_min, route_max = min(history["km"]), max(history["km"])
        if not (route_min - 0.05 <= skm <= route_max + 0.05): continue
        color = "gray" if station["type"] == "X" else ("#0068c9" if station["name"] in stops_set else "#ff4b4b")

        try:
            idx = (np.abs(np.array(history["km"]) - skm)).argmin()

            if x_axis_mode == "Time (MM:SS)":
                x_pos = time_dt_arr[idx]
            else:
                x_pos = dist_arr[idx]

            shapes.append(dict(type="line", xref="x", yref="y", x0=x_pos, x1=x_pos, y0=0, y1=v_max,
                               line=dict(color=color, width=1, dash="dot"), layer="below"))
            shapes.append(dict(type="line", xref="x", yref="y2", x0=x_pos, x1=x_pos, y0=0, y1=e_max,
                               line=dict(color=color, width=1, dash="dot"), layer="below"))

            annotations.append(dict(
                x=x_pos, y=-0.22, xref="x", yref="y domain",
                text=station["name"].title(),
                showarrow=False, font=dict(size=11, color=color),
                xanchor="right", yanchor="top", textangle=-45
            ))

            annotations.append(dict(
                x=x_pos, y=-0.22, xref="x", yref="y2 domain",
                text=station["name"].title(),
                showarrow=False, font=dict(size=11, color=color),
                xanchor="right", yanchor="top", textangle=-45
            ))
        except ValueError:
            pass

    fig.update_layout(
        height=850, margin=dict(l=40, r=40, t=40, b=140), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, title=x_title, range=[x_min, x_max], tickformat=x_format, showticklabels=True),
        xaxis2=dict(showgrid=True, title=x_title, range=[x_min, x_max], tickformat=x_format, showticklabels=True),
        yaxis=dict(title="Speed (km/h)", showgrid=True),
        yaxis2=dict(title="Energy (kWh)", showgrid=True),
        shapes=shapes, annotations=annotations
    )
    return fig


# ==========================================
#  MAIN DASHBOARD
# ==========================================
st.title("🚆 Railway Energy Simulator")

with st.spinner("Loading railML data from ZIP & Building Topological Graph..."):
    nodes, edges = load_railml_from_zip(RAILML_ZIP_PATH)

if not nodes or not edges:
    st.error(f"Failed to load railML topology. Please ensure {RAILML_ZIP_PATH} exists and contains valid railML.")
    st.stop()

st.sidebar.header("📂 Data Source")
st.sidebar.success(f"Network Topological Graph Built!")

# --- 1. ROUTE SELECTION ---
st.sidebar.header("1. Route Selection")

all_line_names = sorted(list(set(e["line_name"] for e in edges if e["line_name"] != "Unknown")))
if not all_line_names:
    st.error("No named lines found in the dataset.")
    st.stop()

selected_lines = st.sidebar.multiselect(
    "Select Railway Line(s)",
    all_line_names,
    default=[all_line_names[0]],
    help="Select one or more lines to combine for your route."
)

if not selected_lines:
    st.warning("Please select at least one railway line to populate the stations.")
    st.stop()

filtered_edges = [e for e in edges if e["line_name"] in selected_lines]
valid_node_ids = set()
for e in filtered_edges:
    valid_node_ids.add(e["u"])
    valid_node_ids.add(e["v"])

filtered_nodes = {nid: n for nid, n in nodes.items() if nid in valid_node_ids}

passenger_stations = [n["name"] for nid, n in filtered_nodes.items() if
                      n["type"] in ["X", "R"] and n["name"] != "Unknown"]
passenger_stations = sorted(list(set(passenger_stations)))

if not passenger_stations:
    st.error("No valid passenger stations found in the selected lines.")
    st.stop()

mc_start = st.sidebar.selectbox("Terminal A", passenger_stations, index=0)
mc_end = st.sidebar.selectbox("Terminal B", passenger_stations, index=len(passenger_stations) - 1)
line_display_name = ", ".join(selected_lines) if len(selected_lines) <= 2 else f"{len(selected_lines)} Selected Lines"

# --- 2. TRAIN PARAMETERS ---
st.sidebar.header("2. Train Parameters")
vehicle_choice = st.sidebar.selectbox("Vehicle Profile", ["Custom"] + list(PREDEFINED_VEHICLES.keys()))

if vehicle_choice == "Custom":
    traction = st.sidebar.selectbox("Traction Type", ["DIESEL", "ELECTRIC"], index=0)
    mass = st.sidebar.number_input("Train Mass (kg)", value=100000, step=10000)
    length = st.sidebar.number_input("Train Length (m)", value=40, step=10)
    power = st.sidebar.number_input("Max Power (kW)", value=600, step=50)
    aux_power = st.sidebar.number_input("Auxiliary Power (kW)", value=40, step=5)
    accel = st.sidebar.slider("Max Acceleration (m/s²)", 0.2, 1.2, 0.6, 0.1)
    decel = st.sidebar.slider("Max Braking (m/s²)", 0.4, 1.5, 0.8, 0.1)
    max_speed = st.sidebar.number_input("Max Speed (km/h)", 30, 350, 160, 10)

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
    max_speed = preset["max_speed"]

    st.sidebar.info(f"**{vehicle_choice} specs loaded:**\n"
                    f"- Max Speed: {max_speed} km/h\n"
                    f"- Traction: {traction}\n"
                    f"- Efficiency: {int(efficiency * 100)}%\n"
                    f"- Mass: {mass:,} kg\n"
                    f"- Power: {power} kW\n"
                    f"- Aux Power: {aux_power} kW")

    if traction == "DIESEL":
        diesel_density = st.sidebar.number_input("Diesel Energy Density (kWh/L)", value=10.0, step=0.1)
    else:
        diesel_density = 10.0

st.sidebar.header("3. Display Options")
x_axis_choice = st.sidebar.radio("X-Axis Display", ["Time (MM:SS)", "Distance (km)"])

# --- 4. ITINERARY BUILDER & MONTE CARLO ---
st.sidebar.header("4. Simulation Mode")
builder_mode = st.sidebar.radio("Mode", ["Manual (Leg-by-Leg)", "Auto-Repeat Round Trip", "Monte Carlo Analysis"])

itinerary_config = []

if builder_mode == "Manual (Leg-by-Leg)":
    num_legs = st.sidebar.number_input("Number of Custom Legs", min_value=1, max_value=15, value=1)
    for i in range(num_legs):
        with st.sidebar.expander(f"Leg {i + 1} Configuration", expanded=(i == 0)):
            if i == 0:
                start_station = st.selectbox(f"Start Station", passenger_stations, key=f"start_{i}")
            else:
                prev_end = itinerary_config[i - 1]["end"]
                st.selectbox(f"Start Station (Locked)", [prev_end], index=0, disabled=True, key=f"start_{i}")
                start_station = prev_end

            default_end = len(passenger_stations) - 1 if i == 0 else 0
            end_station = st.selectbox(f"End Station", passenger_stations, index=default_end, key=f"end_{i}")
            mode = st.selectbox(f"Request Stop Mode", ["random", "all", "none"], key=f"mode_{i}")
            prob = st.slider(f"Probability", 0.0, 1.0, 0.6, 0.05, disabled=(mode != "random"), key=f"prob_{i}")
            dwell = st.number_input(f"Dwell Time (s)", value=30, step=5, key=f"dwell_{i}")

            itinerary_config.append({
                "start": start_station, "end": end_station,
                "mode": mode, "prob": prob, "dwell": dwell
            })

elif builder_mode == "Auto-Repeat Round Trip":
    num_legs = st.sidebar.number_input("Total Number of Legs", min_value=1, max_value=20, value=2)
    mode = st.sidebar.selectbox("Global Stop Mode", ["random", "all", "none"])
    prob = st.sidebar.slider("Global Probability", 0.0, 1.0, 0.6, 0.05, disabled=(mode != "random"))
    dwell = st.sidebar.number_input("Global Dwell Time (s)", value=30, step=5)

    for i in range(num_legs):
        cur_start, cur_end = (mc_start, mc_end) if i % 2 == 0 else (mc_end, mc_start)
        itinerary_config.append({
            "start": cur_start, "end": cur_end,
            "mode": mode, "prob": prob, "dwell": dwell
        })

else:
    # MONTE CARLO MODE
    st.sidebar.caption("Run hundreds of probabilistic variations to find the Expected Value.")
    mc_runs = st.sidebar.number_input("N (Runs per Probability)", min_value=10, max_value=1000, value=100, step=10)
    mc_dwell = st.sidebar.number_input("Dwell Time (s)", value=30, step=5)

# --- RUN BUTTONS ---
if builder_mode in ["Manual (Leg-by-Leg)", "Auto-Repeat Round Trip"]:
    if st.sidebar.button("▶ Run Itinerary", type="primary", use_container_width=True):
        st.session_state.mc_results = None
        st.session_state.journey_results = []
        for idx, leg in enumerate(itinerary_config):
            if leg["start"] == leg["end"]:
                st.sidebar.error(f"Error in Leg {idx + 1}: Start and End stations cannot be the same!")
                st.stop()

        with st.spinner(f"Calculating Physics from topological graph..."):
            try:
                global_time, global_dist = 0.0, 0.0
                for idx, leg_cfg in enumerate(itinerary_config):

                    track = build_route_profile(leg_cfg["start"], leg_cfg["end"], filtered_nodes, filtered_edges,
                                                is_reverse=False)
                    if not track:
                        st.error(f"No route found between {leg_cfg['start']} and {leg_cfg['end']}.")
                        st.stop()

                    start_km = 0.0
                    end_km = track.segments[-1]["km_high"] if track.segments else 0.0

                    sim = TrainSimulator(mass, length, power, aux_power, accel, decel, traction, efficiency, max_speed)

                    history, stops, stats_curr = sim.run_simulation(
                        track, start_km, end_km,
                        leg_cfg["mode"], leg_cfg["prob"], leg_cfg["dwell"],
                        global_start_time=global_time, global_start_dist=global_dist
                    )

                    _, _, stats_all = sim.run_simulation(track, start_km, end_km, "all", 1.0, leg_cfg["dwell"])
                    _, _, stats_none = sim.run_simulation(track, start_km, end_km, "none", 0.0, leg_cfg["dwell"])

                    global_time = history["time_s"][-1]
                    global_dist = history["cum_dist_km"][-1]

                    st.session_state.journey_results.append({
                        "leg_num": idx + 1, "config": leg_cfg, "traction": traction,
                        "efficiency": efficiency, "diesel_density": diesel_density,
                        "history": history, "stops": stops, "stats_curr": stats_curr,
                        "stats_all": stats_all, "stats_none": stats_none, "track": track
                    })
            except Exception as e:
                st.error(f"Error during simulation: {e}")

else:
    if st.sidebar.button("🎲 Run Monte Carlo", type="primary", use_container_width=True):
        st.session_state.journey_results = []
        if mc_start == mc_end:
            st.sidebar.error("Start and End stations cannot be the same!")
            st.stop()

        with st.spinner(f"Running Monte Carlo (N={mc_runs})..."):
            try:
                track = build_route_profile(mc_start, mc_end, filtered_nodes, filtered_edges, is_reverse=False)
                if not track:
                    st.error(f"No valid physical route found between {mc_start} and {mc_end}.")
                    st.stop()

                k_a = 0.0
                k_b = track.segments[-1]["km_high"] if track.segments else 0.0

                sim = TrainSimulator(mass, length, power, aux_power, accel, decel, traction, efficiency, max_speed)


                def get_unit(stats):
                    return stats["net_kwh"] / (diesel_density * efficiency) if traction == "DIESEL" else stats[
                        "net_kwh"]


                _, worst_stops, worst_stats = sim.run_simulation(track, k_a, k_b, "all", 1.0, mc_dwell)
                _, best_stops, best_stats = sim.run_simulation(track, k_a, k_b, "none", 0.0, mc_dwell)

                base_worst_unit = get_unit(worst_stats)
                base_best_unit = get_unit(best_stats)

                results = []
                for p in [0.2, 0.4, 0.6, 0.8]:
                    t_sum, e_sum, s_sum = 0.0, 0.0, 0.0
                    for _ in range(int(mc_runs)):
                        _, r_stops, r_stats = sim.run_simulation(track, k_a, k_b, "random", p, mc_dwell)
                        t_sum += r_stats["journey_time_s"]
                        e_sum += get_unit(r_stats)
                        s_sum += len(r_stops)

                    results.append({
                        "Probability": f"{int(p * 100)}%",
                        "Avg Stops": round(s_sum / mc_runs, 1),
                        "Avg Time (s)": round(t_sum / mc_runs, 0),
                        "Expected Consumed": round(e_sum / mc_runs, 2),
                        "Expected Savings": round(base_worst_unit - (e_sum / mc_runs), 2),
                        "Type": "Stochastic"
                    })

                results.insert(0, {
                    "Probability": "100% (Worst Case Baseline)", "Avg Stops": len(worst_stops),
                    "Avg Time (s)": round(worst_stats["journey_time_s"], 0),
                    "Expected Consumed": round(base_worst_unit, 2), "Expected Savings": 0.0,
                    "Type": "Baseline"
                })
                results.append({
                    "Probability": "0% (Best Case Baseline)", "Avg Stops": len(best_stops),
                    "Avg Time (s)": round(best_stats["journey_time_s"], 0),
                    "Expected Consumed": round(base_best_unit, 2),
                    "Expected Savings": round(base_worst_unit - base_best_unit, 2),
                    "Type": "Baseline"
                })

                df = pd.DataFrame(results)
                st.session_state.mc_results = {
                    "df": df, "start": mc_start, "end": mc_end, "runs": mc_runs,
                    "unit": "Liters" if traction == "DIESEL" else "kWh"
                }
            except Exception as e:
                st.error(f"Error during Monte Carlo: {e}")

# ==========================================
#  MAIN VIEW RENDERER
# ==========================================
if not st.session_state.journey_results and st.session_state.mc_results is None:
    st.info("👈 **Select a railway line, configure your simulation in the sidebar, and click 'Run' to begin.**")

# --- RENDER ITINERARY (Standard Mode) ---
elif st.session_state.journey_results:
    cum_time_s = st.session_state.journey_results[-1]["history"]["time_s"][-1]
    cum_stops = sum(len(res["stops"]) for res in st.session_state.journey_results)

    cum_base_val, cum_best_val, cum_curr_val = 0.0, 0.0, 0.0
    master_traction = st.session_state.journey_results[0]["traction"]
    master_unit = "Liters" if master_traction == "DIESEL" else "kWh"

    for res in st.session_state.journey_results:
        if master_traction == "DIESEL":
            cf = res["diesel_density"] * res["efficiency"]
            cum_base_val += res["stats_all"]["net_kwh"] / cf
            cum_best_val += res["stats_none"]["net_kwh"] / cf
            cum_curr_val += res["stats_curr"]["net_kwh"] / cf
        else:
            cum_base_val += res["stats_all"]["net_kwh"]
            cum_best_val += res["stats_none"]["net_kwh"]
            cum_curr_val += res["stats_curr"]["net_kwh"]

    st.subheader(f"🏁 Cumulative Itinerary Summary ({len(st.session_state.journey_results)} Legs)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Travel Time", format_time(cum_time_s))
    c2.metric("Total Stops Made", cum_stops)

    c3.metric(f"Total Consumed ({master_unit})", f"{cum_curr_val:.1f}")
    c4.metric(f"Total Saved ({master_unit})", f"{cum_base_val - cum_curr_val:.1f}", help="Vs. stopping at all requests")

    st.markdown("---")
    st.subheader("🗺️ Itinerary Breakdown")

    for res in st.session_state.journey_results:
        cfg = res["config"]
        with st.expander(f"Leg {res['leg_num']}: {cfg['start']} ➔ {cfg['end']} (Mode: {cfg['mode'].upper()})",
                         expanded=True):
            m, s = int(res["stats_curr"]["journey_time_s"] // 60), int(res["stats_curr"]["journey_time_s"] % 60)
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Leg Specific Time", f"{m:02d}:{s:02d}")
            sc2.metric("Leg Stops", len(res["stops"]))

            if res["traction"] == "DIESEL":
                cf = res["diesel_density"] * res["efficiency"]
                u_used = res["stats_curr"]["net_kwh"] / cf
                u_saved = (res["stats_all"]["net_kwh"] - res["stats_curr"]["net_kwh"]) / cf
            else:
                u_used = res["stats_curr"]["net_kwh"]
                u_saved = res["stats_all"]["net_kwh"] - res["stats_curr"]["net_kwh"]
            sc3.metric(f"Used / Saved ({master_unit})", f"{u_used:.1f} / {u_saved:.1f}")

            fig = create_plotly_figure(res["history"], res["stops"], res["track"].stations,
                                       res["traction"] == "ELECTRIC", x_axis_choice)
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            st.caption(f"**Stops Made:** {', '.join(res['stops']) if res['stops'] else 'None'}")

# --- RENDER MONTE CARLO (MC Mode) ---
elif st.session_state.mc_results is not None:
    mc = st.session_state.mc_results
    df = mc["df"]
    unit = mc["unit"]

    st.subheader(f"🎲 Monte Carlo Expected Values: {mc['start']} ➔ {mc['end']}")
    st.caption(f"Based on **{mc['runs']}** iterations per probability tier. Topology built from `{RAILML_ZIP_PATH}`.")

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Plotting Expected Values
    chart_df = df[df["Type"] == "Stochastic"].copy()
    fig = px.bar(chart_df, x="Probability", y="Expected Savings",
                 title=f"Expected Savings vs. Probability ({unit})",
                 labels={"Expected Savings": f"Savings vs. All Stops ({unit})",
                         "Probability": "Request Stop Probability"},
                 color="Expected Savings", color_continuous_scale="Blues")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "✅ **Data generation complete.** You can copy the values from the table above directly into your academic manuscript!")
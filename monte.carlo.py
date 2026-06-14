import math
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
        "efficiency": 0.30
    },
    "EDITA+Btax": {
        "traction": "DIESEL", "mass": 42000, "length": 30,
        "power": 250, "aux_power": 25, "accel": 0.4, "decel": 0.8,
        "efficiency": 0.30
    },
    "Regiopanter 3 car (Class 640)": {
        "traction": "ELECTRIC", "mass": 159000, "length": 79.4,
        "power": 2040, "aux_power": 80, "accel": 0.8, "decel": 0.9,
        "efficiency": 0.85
    },
    "Regionova (Class 814)": {
        "traction": "DIESEL", "mass": 39600, "length": 28.44,
        "power": 242, "aux_power": 10, "accel": 0.5, "decel": 0.8,
        "efficiency": 0.30
    },
    "750-7 + 4 cars": {
        "traction": "DIESEL", "mass": 207000, "length": 90.16,
        "power": 1550, "aux_power": 40, "accel": 0.4, "decel": 0.8,
        "efficiency": 0.30
    },
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
# Removed cache_resource so it accurately responds to dynamically filtered subgraphs
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

    def run_simulation(self, track, start_km, end_km, stop_mode, stop_prob, dwell_time, record_history=False):
        total_dist_m = abs(start_km - end_km) * 1000
        direction = -1 if start_km > end_km else 1

        events, stops_km, stops_names = self._build_events(track, start_km, end_km, stop_mode, stop_prob, direction)
        events.sort(key=lambda x: x["km"], reverse=(direction == -1))

        dt = 1.0 if record_history else 2.0
        current_v, distance_covered, total_energy_j, total_regen_j, journey_time_s = 0.0, 0.0, 0.0, 0.0, 0.0
        g_accel = 9.8186

        history = {"time_s": [], "km": [], "cum_dist_km": [], "v_actual": [], "v_limit": [], "energy_kwh": [],
                   "regen_kwh": []} if record_history else None

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
                # If we're effectively stopped at the end of the track, snap it to end
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

            effective_limit_m_s, slope = track.get_effective_limit_and_grad(current_km, rear_km)

            f_gradient = self.mass_kg * g_accel * slope
            effective_decel = max(0.05, self.max_decel + (g_accel * slope))
            max_safe_v = effective_limit_m_s

            # Fast forward event pointer past events we have cleared entirely
            while current_event_idx < total_events:
                e = events[current_event_idx]
                dist_to_e_km = (e["km"] - current_km) * direction

                # If the event is strictly mathematically behind us
                if dist_to_e_km < -0.001:
                    current_event_idx += 1
                else:
                    break

            lookahead_limit = min(current_event_idx + 3, total_events)
            for i in range(current_event_idx, lookahead_limit):
                event = events[i]
                dist_to_event = max(0.0, (event["km"] - current_km) * direction) * 1000

                # Prevent braking to 0 exactly on the event and undershooting
                if event["type"] == "stop":
                    dist_to_event = max(0.0, dist_to_event - 2.0)

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

                    # If within 25 meters of the destination stop marker
                    if -0.01 <= dist_to_e_km <= 0.025:
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

                        # Safely progress the pointer to avoid continuous stopping loops
                        current_event_idx += 1

                        # If this was the final stop, snap to end
                        if abs(current_km - end_km) <= 0.05:
                            distance_covered = total_dist_m

        return history, stops_names, {
            "net_kwh": (total_energy_j - total_regen_j) / 3_600_000.0,
            "journey_time_s": journey_time_s,
        }


def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    h = int(m // 60)
    if h > 0: return f"{h:02d}h {m % 60:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


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
st.markdown("Dedicated tool for calculating stochastic expected values based on railML data profiles.")

with st.spinner("Loading railML data from ZIP & Building Topological Graph..."):
    nodes, edges = load_railml_from_zip(RAILML_ZIP_PATH)

if not nodes or not edges:
    st.error(f"Failed to load railML topology. Please ensure {RAILML_ZIP_PATH} exists and contains valid railML.")
    st.stop()

st.sidebar.header("📂 Data Source")
st.sidebar.success(f"Network Topological Graph Built!")

# --- 1. ROUTE SELECTION (GRAPH PATHFINDING) ---
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

# Dynamically filter the nodes and edges based on selected lines
filtered_edges = [e for e in edges if e["line_name"] in selected_lines]

valid_node_ids = set()
for e in filtered_edges:
    valid_node_ids.add(e["u"])
    valid_node_ids.add(e["v"])

filtered_nodes = {nid: n for nid, n in nodes.items() if nid in valid_node_ids}

# Extract only passenger stations for the dropdown from the FILTERED subset
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

# --- 3. SHIFT CONFIGURATION ---
st.sidebar.header("3. Shift Configuration")
trip_pattern = st.sidebar.radio("Trip Pattern", ["Single Direction (A ➔ B)", "Round Trip (A ➔ B ➔ A)"])
num_cycles = st.sidebar.number_input("Number of Cycles", min_value=1, max_value=20, value=1)

# --- 4. STOCHASTIC SETTINGS ---
st.sidebar.header("4. Stochastic Settings")
mc_runs = st.sidebar.number_input("N (Runs per Probability)", min_value=10, max_value=1000, value=100, step=10)
mc_dwell = st.sidebar.number_input("Dwell Time (s)", value=30, step=5)

# --- 5. REPRESENTATIVE GRAPH ---
st.sidebar.header("5. Representative Graph")
plot_dir = st.sidebar.radio("Plot Direction", [f"{mc_start} ➔ {mc_end}", f"{mc_end} ➔ {mc_start}"])
plot_prob = st.sidebar.slider("Stop Probability for Plot", 0.0, 1.0, 0.4, 0.1)
plot_x_axis = st.sidebar.radio("Plot X-Axis", ["Distance (km)", "Time (MM:SS)"])

c_mc, c_plot = st.sidebar.columns(2)

if c_mc.button("🎲 Run MC", type="primary", use_container_width=True):
    if mc_start == mc_end:
        st.sidebar.error("Start and End stations cannot be the same!")
        st.stop()

    with st.spinner(f"Finding path and running Monte Carlo (N={mc_runs})..."):
        try:
            track_1 = build_route_profile(mc_start, mc_end, filtered_nodes, filtered_edges, is_reverse=False)
            if not track_1:
                st.error(f"No valid physical route found between {mc_start} and {mc_end} on the selected lines.")
                st.stop()

            k_a = 0.0
            k_b = track_1.segments[-1]["km_high"] if track_1.segments else 0.0

            # The reverse trip flips gradients internally so the simulator can run backwards
            track_2 = build_route_profile(mc_start, mc_end, filtered_nodes, filtered_edges, is_reverse=True)

            sim = TrainSimulator(mass, length, power, aux_power, accel, decel, traction, efficiency)


            def get_unit(stats):
                return stats["net_kwh"] / (diesel_density * efficiency) if traction == "DIESEL" else stats["net_kwh"]


            n_fwd = num_cycles
            n_rev = num_cycles if trip_pattern == "Round Trip (A ➔ B ➔ A)" else 0
            total_legs = n_fwd + n_rev

            _, w_stops_1, w_stats_1 = sim.run_simulation(track_1, k_a, k_b, "all", 1.0, mc_dwell, record_history=False)
            _, b_stops_1, b_stats_1 = sim.run_simulation(track_1, k_a, k_b, "none", 0.0, mc_dwell, record_history=False)

            if n_rev > 0:
                _, w_stops_2, w_stats_2 = sim.run_simulation(track_2, k_b, k_a, "all", 1.0, mc_dwell,
                                                             record_history=False)
                _, b_stops_2, b_stats_2 = sim.run_simulation(track_2, k_b, k_a, "none", 0.0, mc_dwell,
                                                             record_history=False)
            else:
                w_stops_2, w_stats_2, b_stops_2, b_stats_2 = [], {"journey_time_s": 0, "net_kwh": 0}, [], {
                    "journey_time_s": 0, "net_kwh": 0}

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
                        _, r_stops, r_stats = sim.run_simulation(track_1, k_a, k_b, "random", p, mc_dwell,
                                                                 record_history=False)
                        t_sum += r_stats["journey_time_s"]
                        e_sum += get_unit(r_stats)
                        s_sum += len(r_stops)
                    for _r in range(n_rev):
                        _, r_stops, r_stats = sim.run_simulation(track_2, k_b, k_a, "random", p, mc_dwell,
                                                                 record_history=False)
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
                                           "line": line_display_name, "runs": mc_runs,
                                           "unit": "Liters" if traction == "DIESEL" else "kWh",
                                           "total_legs": total_legs, "cycles": num_cycles, "pattern": trip_pattern,
                                           "mand_per_leg": round(mand_per_leg, 1), "req_per_leg": round(req_per_leg, 1)}
        except Exception as e:
            st.error(f"Error during Monte Carlo: {e}")

if c_plot.button("📈 Plot Run", use_container_width=True):
    if mc_start == mc_end:
        st.sidebar.error("Start and End stations cannot be the same!")
        st.stop()

    with st.spinner(f"Finding path and generating Representative Run..."):
        try:
            is_primary_dir = plot_dir.startswith(mc_start)
            track = build_route_profile(mc_start, mc_end, filtered_nodes, filtered_edges, is_reverse=not is_primary_dir)

            if not track:
                st.error("No valid route found on the selected lines.")
                st.stop()

            p_start = 0.0
            p_end = track.segments[-1]["km_high"] if track.segments else 0.0

            if not is_primary_dir:
                p_start, p_end = p_end, p_start

            sim = TrainSimulator(mass, length, power, aux_power, accel, decel, traction, efficiency)

            history, stops_names, stats = sim.run_simulation(track, p_start, p_end, "random", plot_prob, mc_dwell,
                                                             record_history=True)
            st.session_state.rep_results = {"history": history, "stops_names": stops_names, "stats": stats,
                                            "start_name": mc_start if is_primary_dir else mc_end,
                                            "end_name": mc_end if is_primary_dir else mc_start, "prob": plot_prob,
                                            "line": line_display_name, "track": track, "traction": traction,
                                            "efficiency": efficiency, "diesel_density": diesel_density}
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

        st.subheader(f"Monte Carlo Expected Values: {mc['start']} ➔ {mc['end']} (Line(s): {mc['line']})")
        st.markdown(
            f"**Configuration:** {mc['pattern']} | **Cycles:** {mc['cycles']} | **Total Legs:** {mc['total_legs']}  \n**Stops Per Leg:** {mc['mand_per_leg']} Mandatory | {mc['req_per_leg']} Request")
        st.dataframe(df.drop(columns=["Prob_Num"]), use_container_width=True, hide_index=True)

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
        st.info("👈 **Configure parameters in Section 5 of the sidebar, then click 'Plot Run'.**")
    else:
        rep = st.session_state.rep_results
        st.subheader(
            f"Representative Run: {rep['start_name']} ➔ {rep['end_name']} (Line(s): {rep['line']}, P = {int(rep['prob'] * 100)}%)")

        m, s = int(rep["stats"]["journey_time_s"] // 60), int(rep["stats"]["journey_time_s"] % 60)
        c1, c2, c3 = st.columns(3)
        c1.metric("Run Time", f"{m:02d}:{s:02d}")
        c2.metric("Stops Made", len(rep["stops_names"]))
        if rep["traction"] == "DIESEL":
            c3.metric("Fuel Consumed", f"{rep['stats']['net_kwh'] / (rep['diesel_density'] * rep['efficiency']):.1f} L")
        else:
            c3.metric("Energy Consumed", f"{rep['stats']['net_kwh']:.1f} kWh")

        fig = create_plotly_figure(rep["history"], rep["stops_names"], rep["track"].stations,
                                   rep["traction"] == "ELECTRIC", plot_x_axis)
        st.plotly_chart(fig, use_container_width=True, theme=None)

        csv_data = pd.DataFrame(rep["history"]).to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Telemetry Data (CSV)", data=csv_data,
                           file_name=f"kinematic_profile_P{int(rep['prob'] * 100)}.csv", mime="text/csv")
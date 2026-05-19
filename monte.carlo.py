import math
import glob
import pandas as pd
import random
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

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
#  DATA LOADER FOR UI DROPDOWNS
# ==========================================
@st.cache_data
def load_mandatory_stations(filepath):
    try:
        df = pd.read_excel(filepath)
        df.columns = df.columns.str.strip()
        df["Zastavení"] = df["Zastavení"].fillna("")
        df["Poloha"] = pd.to_numeric(df["Poloha"], errors="coerce")
        df = df.dropna(subset=["Poloha"])
        df["Poloha"] = df["Poloha"].apply(lambda x: x / 1000.0 if x > 1000 else x)

        stations = []
        for _, row in df.iterrows():
            if str(row["Zastavení"]).strip().upper() == "X":
                name = str(row.get("Dopravní bod", "")).strip()
                if name and name != "nan":
                    stations.append((name, row["Poloha"]))

        return sorted(stations, key=lambda x: x[1], reverse=True)
    except Exception:
        return []


# ==========================================
#  CORE CLASSES
# ==========================================
class TrackProfile:
    def __init__(self, excel_filepath: str, is_forward_direction: bool):
        self.is_forward = is_forward_direction
        self.df_raw = self._load_from_excel(excel_filepath)
        self.segments = self._build_segments()
        self.stations = self._extract_stations()
        self.station_dict = {s["name"]: s["km"] for s in self.stations}

    def _load_from_excel(self, filepath: str) -> pd.DataFrame:
        df = pd.read_excel(filepath)
        df.columns = df.columns.str.strip()
        df["Zastavení"] = df["Zastavení"].fillna("")
        df["Poloha"] = pd.to_numeric(df["Poloha"], errors="coerce")
        df["Rychlost"] = pd.to_numeric(df["Rychlost"], errors="coerce")
        df["Sklon"] = pd.to_numeric(df["Sklon"], errors="coerce")
        df = df.dropna(subset=["Poloha"])
        df["Poloha"] = df["Poloha"].apply(lambda x: x / 1000.0 if x > 1000 else x)
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

            # Record telemetry if requested
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


def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    h = int(m // 60)
    if h > 0:
        return f"{h:02d}h {m % 60:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def create_plotly_figure(history, stops_names, all_stations, is_electric, x_axis_mode):
    # Publication Ready Styling
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

        if total_duration >= 3600:
            x_format = "%H:%M:%S"
            x_title = "Cumulative Time (HH:MM:SS)"
        else:
            x_format = "%M:%S"
            x_title = "Cumulative Time (MM:SS)"
    else:
        x_data = dist_arr
        buffer_km = max(0.5, (dist_arr[-1] - dist_arr[0]) * 0.03)
        x_min = dist_arr[0] - buffer_km
        x_max = dist_arr[-1] + buffer_km
        x_title = "Cumulative Distance (km)"
        x_format = None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15)

    # Distinct line styling for black & white or standard print reading
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

        # Grey for mandatory, Black/Blue for executed stops, Red for skipped stops (easier to read)
        color = "gray" if station["type"] == "X" else ("black" if station["name"] in stops_set else "#d62728")

        try:
            idx = (np.abs(np.array(history["km"]) - skm)).argmin()
            x_pos = time_dt_arr[idx] if x_axis_mode == "Time (MM:SS)" else dist_arr[idx]

            shapes.append(dict(type="line", xref="x", yref="y", x0=x_pos, x1=x_pos, y0=0, y1=v_max,
                               line=dict(color=color, width=1, dash="dot"), layer="below"))
            shapes.append(dict(type="line", xref="x", yref="y2", x0=x_pos, x1=x_pos, y0=0, y1=e_max,
                               line=dict(color=color, width=1, dash="dot"), layer="below"))

            annotations.append(
                dict(x=x_pos, y=-0.15, xref="x", yref="y domain", text=station["name"].title(), showarrow=False,
                     font=dict(family="Times New Roman, serif", size=12, color=color), xanchor="right", yanchor="top",
                     textangle=-45))
            annotations.append(
                dict(x=x_pos, y=-0.15, xref="x", yref="y2 domain", text=station["name"].title(), showarrow=False,
                     font=dict(family="Times New Roman, serif", size=12, color=color), xanchor="right", yanchor="top",
                     textangle=-45))
        except ValueError:
            pass

    # Academic publication theme layout
    fig.update_layout(
        template="simple_white",
        font=dict(family="Times New Roman, serif", size=14, color="black"),
        height=800, margin=dict(l=60, r=40, t=40, b=150), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor='lightgray', title=x_title, range=[x_min, x_max], tickformat=x_format,
                   showticklabels=True, linecolor='black', mirror=True),
        xaxis2=dict(showgrid=True, gridcolor='lightgray', title=x_title, range=[x_min, x_max], tickformat=x_format,
                    showticklabels=True, linecolor='black', mirror=True),
        yaxis=dict(title="Speed (km/h)", showgrid=True, gridcolor='lightgray', linecolor='black', mirror=True),
        yaxis2=dict(title="Energy (kWh)", showgrid=True, gridcolor='lightgray', linecolor='black', mirror=True),
        shapes=shapes, annotations=annotations
    )
    return fig


# ==========================================
#  MAIN DASHBOARD
# ==========================================
st.title("🎲 Monte Carlo Fleet Simulator")
st.markdown(
    "Dedicated tool for calculating stochastic expected values and rendering representative kinematic profiles.")

# --- 0. FILE LOADER WIDGET ---
st.sidebar.header("📂 Data Source")
excel_files = [f for f in glob.glob("*.xlsx") if not f.startswith("~$")]
if not excel_files:
    st.sidebar.error("No Excel (.xlsx) files found in the current directory.")
    st.stop()

selected_track_file = st.sidebar.selectbox("Select Track Profile", excel_files)

# Load stations from the dynamically selected file
mandatory_stations = load_mandatory_stations(selected_track_file)
if not mandatory_stations:
    st.error(f"Could not load stations from {selected_track_file}. Ensure it contains 'X' in the Zastavení column.")
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

    if traction == "DIESEL":
        diesel_density = st.sidebar.number_input("Diesel Energy Density (kWh/L)", value=10.0, step=0.1)
    else:
        diesel_density = 10.0

# --- 2. SHIFT & TRIP BUILDER ---
st.sidebar.header("2. Shift Configuration")
mc_start = st.sidebar.selectbox("Terminal A", station_names, index=0)
mc_end = st.sidebar.selectbox("Terminal B", station_names, index=len(station_names) - 1)

trip_pattern = st.sidebar.radio("Trip Pattern", ["Single Direction (A ➔ B)", "Round Trip (A ➔ B ➔ A)"])
num_cycles = st.sidebar.number_input("Number of Cycles", min_value=1, max_value=20, value=1,
                                     help="How many times the selected pattern is repeated in a single shift.")

st.sidebar.header("3. Stochastic Settings")
mc_runs = st.sidebar.number_input("N (Runs per Probability)", min_value=10, max_value=1000, value=100, step=10,
                                  help="The 'N' variable for the Monte Carlo loops.")
mc_dwell = st.sidebar.number_input("Dwell Time (s)", value=30, step=5)

st.sidebar.header("4. Representative Graph")
st.sidebar.caption("Visualize a single simulated run to view velocity, limits, and energy profiles.")
plot_dir = st.sidebar.radio("Plot Direction", [f"{mc_start} ➔ {mc_end}", f"{mc_end} ➔ {mc_start}"])
plot_prob = st.sidebar.slider("Stop Probability for Plot", 0.0, 1.0, 0.4, 0.1)
plot_x_axis = st.sidebar.radio("Plot X-Axis", ["Distance (km)", "Time (MM:SS)"])

# --- RUN BUTTONS ---
c_mc, c_plot = st.sidebar.columns(2)
if c_mc.button("🎲 Run MC", type="primary", use_container_width=True):
    if mc_start == mc_end:
        st.sidebar.error("Start and End stations cannot be the same!")
        st.stop()

    with st.spinner(f"Running Monte Carlo (N={mc_runs}) using {selected_track_file}..."):
        try:
            # Set up directional track files to account for asymmetric gradients
            is_fwd_1 = station_dict[mc_start] > station_dict[mc_end]
            track_1 = TrackProfile(selected_track_file, is_forward_direction=is_fwd_1)

            is_fwd_2 = station_dict[mc_end] > station_dict[mc_start]
            track_2 = TrackProfile(selected_track_file, is_forward_direction=is_fwd_2)

            sim = TrainSimulator(mass, length, power, aux_power, accel, decel, traction, efficiency)


            def get_unit(stats):
                return stats["net_kwh"] / (diesel_density * efficiency) if traction == "DIESEL" else stats["net_kwh"]


            # Shift Logic Variables
            n_fwd = num_cycles
            n_rev = num_cycles if trip_pattern == "Round Trip (A ➔ B ➔ A)" else 0

            # Deterministic Baselines (Fast, no history tracking)
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

            # Scale baselines
            base_worst_unit = (get_unit(w_stats_1) * n_fwd) + (get_unit(w_stats_2) * n_rev)
            base_best_unit = (get_unit(b_stats_1) * n_fwd) + (get_unit(b_stats_2) * n_rev)
            base_worst_time = (w_stats_1["journey_time_s"] * n_fwd) + (w_stats_2["journey_time_s"] * n_rev)
            base_best_time = (b_stats_1["journey_time_s"] * n_fwd) + (b_stats_2["journey_time_s"] * n_rev)
            base_worst_stops = (len(w_stops_1) * n_fwd) + (len(w_stops_2) * n_rev)
            base_best_stops = (len(b_stops_1) * n_fwd) + (len(b_stops_2) * n_rev)

            # Stochastic Loops (Fast, no history tracking)
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

                results.append({
                    "Probability": f"{int(p * 100)}%",
                    "Avg Stops": round(s_sum / mc_runs, 1),
                    "Avg Time": format_time(t_sum / mc_runs),
                    "Expected Consumed": round(e_sum / mc_runs, 2),
                    "Expected Savings": round(base_worst_unit - (e_sum / mc_runs), 2),
                    "Type": "Stochastic"
                })

            results.insert(0, {
                "Probability": "100% (Worst Case)", "Avg Stops": base_worst_stops,
                "Avg Time": format_time(base_worst_time),
                "Expected Consumed": round(base_worst_unit, 2), "Expected Savings": 0.0,
                "Type": "Baseline"
            })
            results.append({
                "Probability": "0% (Best Case)", "Avg Stops": base_best_stops,
                "Avg Time": format_time(base_best_time),
                "Expected Consumed": round(base_best_unit, 2),
                "Expected Savings": round(base_worst_unit - base_best_unit, 2),
                "Type": "Baseline"
            })

            st.session_state.mc_results = {
                "df": pd.DataFrame(results), "start": mc_start, "end": mc_end, "runs": mc_runs,
                "unit": "Liters" if traction == "DIESEL" else "kWh",
                "total_legs": n_fwd + n_rev, "cycles": num_cycles, "pattern": trip_pattern
            }
        except Exception as e:
            st.error(f"Error during Monte Carlo: {e}")

if c_plot.button("📈 Plot Run", use_container_width=True):
    if mc_start == mc_end:
        st.sidebar.error("Start and End stations cannot be the same!")
        st.stop()

    with st.spinner("Generating Representative Run telemetry..."):
        try:
            # Parse requested direction
            is_primary_dir = plot_dir.startswith(mc_start)
            p_start = station_dict[mc_start] if is_primary_dir else station_dict[mc_end]
            p_end = station_dict[mc_end] if is_primary_dir else station_dict[mc_start]

            is_fwd = p_start > p_end
            track = TrackProfile(selected_track_file, is_forward_direction=is_fwd)
            sim = TrainSimulator(mass, length, power, aux_power, accel, decel, traction, efficiency)

            # Execute exactly one run WITH history recording turned ON
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
            f"**Configuration:** {mc['pattern']} | **Cycles:** {mc['cycles']} | **Total Legs:** {mc['total_legs']}")
        st.caption(
            f"Based on **{mc['runs']}** iterations per probability tier. Data loaded from **{selected_track_file}**.")

        st.dataframe(df, use_container_width=True, hide_index=True)

        chart_df = df[df["Type"] == "Stochastic"].copy()
        fig = px.bar(chart_df, x="Probability", y="Expected Savings",
                     title=f"Expected Savings vs. Probability ({unit})",
                     labels={"Expected Savings": f"Savings vs. All Stops ({unit})",
                             "Probability": "Request Stop Probability"},
                     color="Expected Savings", color_continuous_scale="Blues")
        fig.update_layout(showlegend=False)
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

        # Override Streamlit's theme to force the white academic background to show
        st.plotly_chart(fig, use_container_width=True, theme=None)

        st.caption(
            f"**Stops Made during this specific run:** {', '.join(rep['stops_names']) if rep['stops_names'] else 'None'}")

        st.markdown("---")
        st.subheader("📥 Export for Publication")
        st.write(
            "You can use the built-in camera icon in the top right of the graph to download a PNG. Alternatively, download the raw telemetry data below to plot the graph natively in LaTeX (using `pgfplots`) or Excel.")

        csv_df = pd.DataFrame(rep["history"])
        csv_data = csv_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Telemetry Data (CSV)",
            data=csv_data,
            file_name=f"kinematic_profile_P{int(rep['prob'] * 100)}.csv",
            mime="text/csv",
            help="Download the raw second-by-second physics data to create high-resolution plots in LaTeX."
        )
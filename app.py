import math
import os
import pandas as pd
import random
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ==========================================
#  PRE-DEFINED VEHICLE FLEET
# ==========================================
# You can add or edit any trains here! They will automatically appear in the UI dropdown.
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
}

# ==========================================
#  STREAMLIT PAGE CONFIG & STATE
# ==========================================
st.set_page_config(page_title="Railway Energy Simulator", layout="wide")

if "journey_results" not in st.session_state:
    st.session_state.journey_results = []


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

    def run_simulation(self, track, start_km, end_km, stop_mode, stop_prob, dwell_time):
        total_dist_m = abs(start_km - end_km) * 1000
        direction = -1 if start_km > end_km else 1
        events, stops_km, stops_names = self._build_events(track, start_km, end_km, stop_mode, stop_prob, direction)

        dt, current_v, distance_covered, total_energy_j, total_regen_j, journey_time_s = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
        history = {"time_s": [], "km": [], "v_actual": [], "v_limit": [], "energy_kwh": [], "regen_kwh": []}

        g_accel = 9.8186

        while distance_covered < total_dist_m:
            current_km = start_km + (distance_covered / 1000.0) * direction
            rear_km = current_km - (self.train_length_m / 1000.0) * direction
            current_limit_m_s, slope = track.get_effective_limit_and_grad(current_km, rear_km)

            f_gradient = self.mass_kg * g_accel * slope
            effective_decel = max(0.05, self.max_decel + (g_accel * slope))
            max_safe_v = current_limit_m_s

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

            if current_v > max_safe_v + 1e-4:
                f_resist = self.get_resistance(current_v)
                natural_decel = (f_resist + f_gradient) / self.eff_mass
                required_decel = (current_v - max_safe_v) / dt
                brake_application = min(self.max_decel, required_decel - natural_decel)
                brake_application = max(0.0, brake_application)
                actual_decel = brake_application + natural_decel

                regen_power = (self.eff_mass * brake_application) * current_v * self.regen_efficiency
                current_v = max(0.0, current_v - actual_decel * dt)

            elif current_v < min(current_limit_m_s, max_safe_v) - 1e-4:
                f_resist = self.get_resistance(current_v)
                total_desired_force = f_resist + (self.eff_mass * self.max_accel) + f_gradient
                actual_force = max(0.0, min(total_desired_force, self.max_power_w / max(current_v, 0.5)))
                current_v = max(0.0, min(current_v + ((actual_force - f_resist - f_gradient) / self.eff_mass) * dt,
                                         current_limit_m_s, max_safe_v))
                mech_power = max(0.0, actual_force * current_v)
            else:
                f_resist = self.get_resistance(current_v)
                total_desired_force = f_resist + f_gradient
                actual_force = max(0.0, min(total_desired_force, self.max_power_w / max(current_v, 0.5)))
                current_v = max(0.0, current_v + ((actual_force - f_resist - f_gradient) / self.eff_mass) * dt)
                mech_power = max(0.0, actual_force * current_v)

            history["time_s"].append(journey_time_s)
            history["km"].append(current_km)
            history["v_actual"].append(current_v * 3.6)
            history["v_limit"].append(current_limit_m_s * 3.6)
            history["energy_kwh"].append(total_energy_j / 3_600_000.0)
            history["regen_kwh"].append(total_regen_j / 3_600_000.0)

            total_energy_j += ((mech_power / self.traction_efficiency) + self.aux_power_w) * dt
            total_regen_j += regen_power * dt
            distance_covered += current_v * dt
            journey_time_s += dt

            if current_v < 0.5 and any(abs(current_km - s) <= 0.05 for s in stops_km):
                current_v = 0.0
                for _ in range(2):
                    history["time_s"].append(journey_time_s)
                    history["km"].append(current_km)
                    history["v_actual"].append(0.0)
                    history["v_limit"].append(current_limit_m_s * 3.6)
                    history["energy_kwh"].append(total_energy_j / 3_600_000.0)
                    history["regen_kwh"].append(total_regen_j / 3_600_000.0)
                    if _ == 0:
                        total_energy_j += self.aux_power_w * dwell_time
                        journey_time_s += dwell_time

                events = [e for e in events if not (e["type"] == "stop" and abs(e["km"] - current_km) <= 0.05)]
                stops_km = [s for s in stops_km if abs(s - current_km) > 0.05]

        return history, stops_names, {
            "total_kwh": total_energy_j / 3_600_000.0,
            "regen_kwh": total_regen_j / 3_600_000.0,
            "net_kwh": (total_energy_j - total_regen_j) / 3_600_000.0,
            "journey_time_s": journey_time_s,
        }


# ==========================================
#  UI & PLOTTING HELPERS
# ==========================================
def create_plotly_figure(history, stops_names, all_stations, is_electric, x_axis_mode):
    base_time = pd.to_datetime("1970-01-01")
    time_dt_arr = [base_time + pd.to_timedelta(s, unit='s') for s in history["time_s"]]
    dist_arr = history["km"]
    stops_set = set(stops_names)
    net = [g - r for g, r in zip(history["energy_kwh"], history["regen_kwh"])]

    if x_axis_mode == "Time (MM:SS)":
        x_data = time_dt_arr
        total_duration = history["time_s"][-1]
        buffer_seconds = max(30, int(total_duration * 0.03))
        x_min = time_dt_arr[0] - pd.Timedelta(seconds=buffer_seconds)
        x_max = time_dt_arr[-1] + pd.Timedelta(seconds=buffer_seconds)
        x_title = "Time (MM:SS)"
        x_format = "%M:%S"
    else:
        x_data = dist_arr
        route_min, route_max = min(dist_arr), max(dist_arr)
        buffer_km = max(0.5, (route_max - route_min) * 0.03)
        if dist_arr[0] > dist_arr[-1]:
            x_min = dist_arr[0] + buffer_km
            x_max = dist_arr[-1] - buffer_km
        else:
            x_min = dist_arr[0] - buffer_km
            x_max = dist_arr[-1] + buffer_km
        x_title = "Distance (km)"
        x_format = None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    fig.add_trace(
        go.Scatter(x=x_data, y=history["v_limit"], name="Speed Limit", line=dict(color="#ff4b4b", dash="dash")), row=1,
        col=1)
    fig.add_trace(go.Scatter(x=x_data, y=history["v_actual"], name="Actual Speed", line=dict(color="#0068c9", width=2),
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
            if x_axis_mode == "Time (MM:SS)":
                idx = (np.abs(np.array(history["km"]) - skm)).argmin()
                x_pos = time_dt_arr[idx]
            else:
                x_pos = skm

            shapes.append(dict(type="line", xref="x", yref="y", x0=x_pos, x1=x_pos, y0=0, y1=v_max,
                               line=dict(color=color, width=1, dash="dot"), layer="below"))
            shapes.append(dict(type="line", xref="x", yref="y2", x0=x_pos, x1=x_pos, y0=0, y1=e_max,
                               line=dict(color=color, width=1, dash="dot"), layer="below"))

            annotations.append(dict(
                x=x_pos, y=0.03, xref="x", yref="y domain",
                text=f" {station['name'].title()} ",
                showarrow=False, font=dict(size=11, color=color),
                xanchor="center", yanchor="bottom",
                bgcolor="rgba(0,0,0,0.4)"
            ))
        except ValueError:
            pass

    fig.update_layout(
        height=750, margin=dict(l=40, r=40, t=40, b=40), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, title=x_title, range=[x_min, x_max], tickformat=x_format),
        xaxis2=dict(showgrid=True, title=x_title, range=[x_min, x_max], tickformat=x_format),
        yaxis=dict(title="Speed (km/h)", showgrid=True),
        yaxis2=dict(title="Energy (kWh)", showgrid=True),
        shapes=shapes, annotations=annotations
    )
    return fig


def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    h = int(m // 60)
    if h > 0:
        return f"{h}:{m % 60:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ==========================================
#  MAIN DASHBOARD
# ==========================================
st.title("🚆 Railway Energy Simulator")

mandatory_stations = load_mandatory_stations("track_profile.xlsx")
if not mandatory_stations:
    st.error(
        "Could not load stations. Ensure 'track_profile.xlsx' is in the folder and contains 'X' in the Zastavení column.")
    st.stop()

station_names = [s[0] for s in mandatory_stations]
station_dict = {s[0]: s[1] for s in mandatory_stations}

# --- SIDEBAR WIDGETS ---
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

    st.sidebar.info(f"**{vehicle_choice} specs loaded:**\n"
                    f"- Traction: {traction}\n"
                    f"- Efficiency: {int(efficiency * 100)}%\n"
                    f"- Mass: {mass:,} kg\n"
                    f"- Power: {power} kW\n"
                    f"- Aux Power: {aux_power} kW")

    if traction == "DIESEL":
        diesel_density = st.sidebar.number_input("Diesel Energy Density (kWh/L)", value=10.0, step=0.1)
    else:
        diesel_density = 10.0

st.sidebar.header("2. Display Options")
x_axis_choice = st.sidebar.radio("X-Axis Display", ["Time (MM:SS)", "Distance (km)"])

# --- ITINERARY BUILDER ---
st.sidebar.header("3. Itinerary Builder")
builder_mode = st.sidebar.radio("Builder Mode", ["Manual (Leg-by-Leg)", "Auto-Repeat Round Trip"])

itinerary_config = []

if builder_mode == "Manual (Leg-by-Leg)":
    num_legs = st.sidebar.number_input("Number of Custom Legs", min_value=1, max_value=15, value=1)

    for i in range(num_legs):
        with st.sidebar.expander(f"Leg {i + 1} Configuration", expanded=(i == 0)):
            if i == 0:
                start_station = st.selectbox(f"Start Station", station_names, key=f"start_{i}")
            else:
                # ENFORCED DEPENDENCY: Lock to previous end station
                prev_end = itinerary_config[i - 1]["end"]
                start_station = st.selectbox(f"Start Station (Locked)", station_names,
                                             index=station_names.index(prev_end), disabled=True, key=f"start_{i}")

            default_end = len(station_names) - 1 if i == 0 else 0
            end_station = st.selectbox(f"End Station", station_names, index=default_end, key=f"end_{i}")

            mode = st.selectbox(f"Request Stop Mode", ["random", "all", "none"], key=f"mode_{i}")
            prob = st.slider(f"Probability", 0.0, 1.0, 0.6, 0.05, disabled=(mode != "random"), key=f"prob_{i}")
            dwell = st.number_input(f"Dwell Time (s)", value=30, step=5, key=f"dwell_{i}")

            itinerary_config.append({
                "start": start_station, "start_km": station_dict[start_station],
                "end": end_station, "end_km": station_dict[end_station],
                "mode": mode, "prob": prob, "dwell": dwell
            })

else:  # Auto-Repeat Round Trip Mode
    st.sidebar.caption("Automatically generates back-and-forth legs.")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        base_start = st.selectbox("Terminal A", station_names, index=0)
    with col2:
        base_end = st.selectbox("Terminal B", station_names, index=len(station_names) - 1)

    num_legs = st.sidebar.number_input("Total Number of Legs", min_value=1, max_value=20, value=2,
                                       help="e.g., 2 legs = 1 full round trip")

    st.sidebar.markdown("---")
    mode = st.sidebar.selectbox("Global Stop Mode", ["random", "all", "none"])
    prob = st.sidebar.slider("Global Probability", 0.0, 1.0, 0.6, 0.05, disabled=(mode != "random"))
    dwell = st.sidebar.number_input("Global Dwell Time (s)", value=30, step=5)

    for i in range(num_legs):
        # Even legs (0, 2, 4) go A -> B. Odd legs (1, 3, 5) go B -> A.
        if i % 2 == 0:
            cur_start, cur_end = base_start, base_end
        else:
            cur_start, cur_end = base_end, base_start

        itinerary_config.append({
            "start": cur_start, "start_km": station_dict[cur_start],
            "end": cur_end, "end_km": station_dict[cur_end],
            "mode": mode, "prob": prob, "dwell": dwell
        })

# --- RUN BUTTON ---
if st.sidebar.button("▶ Run Full Itinerary", type="primary", use_container_width=True):
    for idx, leg in enumerate(itinerary_config):
        if leg["start_km"] == leg["end_km"]:
            st.sidebar.error(f"Error in Leg {idx + 1}: Start and End stations cannot be the same!")
            st.stop()

    st.session_state.journey_results = []

    with st.spinner("Calculating Physics for Full Itinerary..."):
        try:
            for idx, leg_cfg in enumerate(itinerary_config):
                is_fwd = leg_cfg["start_km"] > leg_cfg["end_km"]
                track = TrackProfile("track_profile.xlsx", is_forward_direction=is_fwd)
                sim = TrainSimulator(mass, length, power, aux_power, accel, decel, traction, efficiency)

                history, stops, stats_curr = sim.run_simulation(track, leg_cfg["start_km"], leg_cfg["end_km"],
                                                                leg_cfg["mode"], leg_cfg["prob"], leg_cfg["dwell"])
                _, _, stats_all = sim.run_simulation(track, leg_cfg["start_km"], leg_cfg["end_km"], "all", 1.0,
                                                     leg_cfg["dwell"])
                _, _, stats_none = sim.run_simulation(track, leg_cfg["start_km"], leg_cfg["end_km"], "none", 0.0,
                                                      leg_cfg["dwell"])

                st.session_state.journey_results.append({
                    "leg_num": idx + 1,
                    "config": leg_cfg,
                    "traction": traction,
                    "efficiency": efficiency,
                    "diesel_density": diesel_density,
                    "history": history,
                    "stops": stops,
                    "stats_curr": stats_curr,
                    "stats_all": stats_all,
                    "stats_none": stats_none
                })
        except Exception as e:
            st.error(f"Error during simulation: {e}")

# ==========================================
#  MAIN VIEW RENDERER
# ==========================================
if not st.session_state.journey_results:
    st.info("👈 **Configure your multi-leg itinerary in the sidebar and click 'Run Full Itinerary' to begin.**")
else:
    # --- CUMULATIVE MATH ---
    cum_time_s = sum(res["stats_curr"]["journey_time_s"] for res in st.session_state.journey_results)
    cum_stops = sum(len(res["stops"]) for res in st.session_state.journey_results)

    cum_liters_curr, cum_liters_all = 0.0, 0.0
    cum_kwh_curr, cum_kwh_all = 0.0, 0.0

    for res in st.session_state.journey_results:
        if res["traction"] == "DIESEL":
            conversion_factor = res["diesel_density"] * res["efficiency"]
            cum_liters_curr += res["stats_curr"]["net_kwh"] / conversion_factor
            cum_liters_all += res["stats_all"]["net_kwh"] / conversion_factor
        else:
            cum_kwh_curr += res["stats_curr"]["net_kwh"]
            cum_kwh_all += res["stats_all"]["net_kwh"]

    # --- TOP DASHBOARD ---
    st.subheader(f"🏁 Cumulative Itinerary Summary ({len(st.session_state.journey_results)} Legs)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Travel Time", format_time(cum_time_s))
    c2.metric("Total Stops Made", cum_stops)

    if cum_liters_curr > 0:
        c3.metric("Total Fuel Consumed", f"{cum_liters_curr:.1f} L")
        c4.metric("Total Fuel Saved", f"{cum_liters_all - cum_liters_curr:.1f} L", help="Vs. stopping at all requests")
    elif cum_kwh_curr > 0:
        c3.metric("Total Net Energy", f"{cum_kwh_curr:.1f} kWh")
        c4.metric("Total Energy Saved", f"{cum_kwh_all - cum_kwh_curr:.1f} kWh", help="Vs. stopping at all requests")

    st.markdown("---")

    # --- INDIVIDUAL LEG BREAKDOWNS ---
    st.subheader("🗺️ Itinerary Breakdown")

    for res in st.session_state.journey_results:
        cfg = res["config"]

        with st.expander(f"Leg {res['leg_num']}: {cfg['start']} ➔ {cfg['end']} (Mode: {cfg['mode'].upper()})",
                         expanded=True):

            m = int(res["stats_curr"]["journey_time_s"] // 60)
            s = int(res["stats_curr"]["journey_time_s"] % 60)

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Leg Time", f"{m:02d}:{s:02d}")
            sc2.metric("Leg Stops", len(res["stops"]))

            if res["traction"] == "DIESEL":
                conversion_factor = res["diesel_density"] * res["efficiency"]
                l_used = res["stats_curr"]["net_kwh"] / conversion_factor
                l_saved = (res["stats_all"]["net_kwh"] - res["stats_curr"]["net_kwh"]) / conversion_factor
                sc3.metric("Fuel Used / Saved", f"{l_used:.1f} L / {l_saved:.1f} L")
            else:
                k_used = res["stats_curr"]["net_kwh"]
                k_saved = res["stats_all"]["net_kwh"] - res["stats_curr"]["net_kwh"]
                sc3.metric("Energy Used / Saved", f"{k_used:.1f} kWh / {k_saved:.1f} kWh")

            fig = create_plotly_figure(res["history"], res["stops"], TrackProfile("track_profile.xlsx", True).stations,
                                       res["traction"] == "ELECTRIC", x_axis_choice)
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

            st.caption(f"**Stops Made:** {', '.join(res['stops']) if res['stops'] else 'None'}")

            route_min = min(cfg["start_km"], cfg["end_km"])
            route_max = max(cfg["start_km"], cfg["end_km"])
            all_rq = [s["name"] for s in TrackProfile("track_profile.xlsx", True).stations if
                      s["type"] == "R" and route_min <= s["km"] <= route_max]
            skipped = [name for name in all_rq if name not in res["stops"]]
            st.caption(f"**Stops Skipped:** {', '.join(skipped) if skipped else 'None'}")
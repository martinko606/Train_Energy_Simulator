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
        "traction": "DIESEL", "mass": 20000, "length": 15,
        "power": 115, "aux_power": 10, "accel": 0.5, "decel": 0.8
    },
    "EDITA+Btax": {
        "traction": "DIESEL", "mass": 35000, "length": 30,
        "power": 115, "aux_power": 10, "accel": 0.4, "decel": 0.8
    },
}

# ==========================================
#  STREAMLIT PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Railway Energy Simulator", layout="wide")


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
    def __init__(self, mass_kg, length_m, max_power_kw, aux_power_kw, max_accel, max_decel, traction_type):
        self.mass_kg = mass_kg
        self.eff_mass = self.mass_kg * 1.08
        self.A, self.B, self.C = 1500, 30, 4
        self.traction_efficiency = 0.85
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

        while distance_covered < total_dist_m:
            current_km = start_km + (distance_covered / 1000.0) * direction
            rear_km = current_km - (self.train_length_m / 1000.0) * direction
            current_limit_m_s, slope = track.get_effective_limit_and_grad(current_km, rear_km)

            f_gradient, max_safe_v = self.mass_kg * 9.81 * slope, current_limit_m_s

            for event in events:
                tolerance = 0.05 if event["type"] == "stop" else 1e-4
                is_ahead = (event["km"] <= current_km + tolerance) if direction == -1 else (
                            event["km"] >= current_km - tolerance)
                overshot = (current_km < event["km"]) if direction == -1 else (current_km > event["km"])

                if not is_ahead: continue

                dist_to_event = 0.0 if (event["type"] == "stop" and overshot) else abs(current_km - event["km"]) * 1000

                if event["target_v"] < current_v:
                    safe_v = math.sqrt(max(0.0, event["target_v"] ** 2 + 2 * self.max_decel * dist_to_event))
                    max_safe_v = min(max_safe_v, safe_v)

            mech_power, regen_power = 0.0, 0.0

            if current_v > max_safe_v + 1e-4:
                decel = min(self.max_decel, (current_v - max_safe_v) / dt)
                regen_power = (self.eff_mass * decel) * current_v * self.regen_efficiency
                current_v = max(0.0, current_v - decel * dt)
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
def create_plotly_figure(history, stops_names, all_stations, is_electric):
    base_time = pd.to_datetime("1970-01-01")
    time_dt_arr = [base_time + pd.to_timedelta(s, unit='s') for s in history["time_s"]]
    stops_set = set(stops_names)
    net = [g - r for g, r in zip(history["energy_kwh"], history["regen_kwh"])]

    # --- Add margin/buffer to X-Axis to prevent clipping at the start and end ---
    total_duration = history["time_s"][-1]
    buffer_seconds = max(30, int(total_duration * 0.02))  # Add 2% padding, minimum 30 seconds
    t_min = time_dt_arr[0] - pd.Timedelta(seconds=buffer_seconds)
    t_max = time_dt_arr[-1] + pd.Timedelta(seconds=buffer_seconds)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    fig.add_trace(
        go.Scatter(x=time_dt_arr, y=history["v_limit"], name="Speed Limit", line=dict(color="#f85149", dash="dash")),
        row=1, col=1)
    fig.add_trace(
        go.Scatter(x=time_dt_arr, y=history["v_actual"], name="Actual Speed", line=dict(color="#3fb950", width=2),
                   fill="tozeroy", fillcolor="rgba(63,185,80,0.1)"), row=1, col=1)

    fig.add_trace(
        go.Scatter(x=time_dt_arr, y=history["energy_kwh"], name="Gross Energy", line=dict(color="#d29922", width=2),
                   fill="tozeroy", fillcolor="rgba(210,153,34,0.1)"), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=time_dt_arr, y=net, name="Net Energy", line=dict(color="#bc8cff", width=2, dash="dashdot")), row=2,
        col=1)

    if is_electric:
        fig.add_trace(go.Scatter(x=time_dt_arr, y=history["regen_kwh"], name="Regen Recovered",
                                 line=dict(color="#58a6ff", width=1.5, dash="dot"), fill="tozeroy",
                                 fillcolor="rgba(88,166,255,0.1)"), row=2, col=1)

    shapes, annotations = [], []
    v_max = max(max(history["v_limit"]), max(history["v_actual"])) * 1.05
    e_max = max(history["energy_kwh"]) * 1.05

    for station in all_stations:
        if station["type"] not in ["X", "R"]: continue
        skm = station["km"]
        color = "#ffffff" if station["type"] == "X" else ("#79c0ff" if station["name"] in stops_set else "#484f58")
        try:
            idx = (np.abs(np.array(history["km"]) - skm)).argmin()
            t_dt = time_dt_arr[idx]

            shapes.append(dict(type="line", xref="x", yref="y", x0=t_dt, x1=t_dt, y0=0, y1=v_max,
                               line=dict(color=color, width=1, dash="dot"), layer="below"))
            shapes.append(dict(type="line", xref="x", yref="y2", x0=t_dt, x1=t_dt, y0=0, y1=e_max,
                               line=dict(color=color, width=1, dash="dot"), layer="below"))

            annotations.append(dict(
                x=t_dt, y=-0.06, xref="x", yref="y domain",
                text=station["name"].title(),
                showarrow=False, font=dict(size=11, color=color),
                xanchor="center", yanchor="top"
            ))
        except ValueError:
            pass

    fig.update_layout(
        height=700, margin=dict(l=40, r=40, t=40, b=80), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor="#30363d", tickformat="%M:%S", range=[t_min, t_max]),
        xaxis2=dict(showgrid=True, gridcolor="#30363d", title="Time (MM:SS)", tickformat="%M:%S", range=[t_min, t_max]),
        yaxis=dict(title="Speed (km/h)", showgrid=True, gridcolor="#30363d"),
        yaxis2=dict(title="Energy (kWh)", showgrid=True, gridcolor="#30363d"),
        shapes=shapes, annotations=annotations
    )
    return fig


# ==========================================
#  MAIN DASHBOARD
# ==========================================
st.title("Railway Energy Simulator")

mandatory_stations = load_mandatory_stations("track_profile.xlsx")
if not mandatory_stations:
    st.error(
        "Could not load stations. Ensure 'track_profile.xlsx' is in the folder and contains 'X' in the Zastavení column.")
    st.stop()

station_names = [s[0] for s in mandatory_stations]
station_dict = {s[0]: s[1] for s in mandatory_stations}

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
else:
    preset = PREDEFINED_VEHICLES[vehicle_choice]
    traction = preset["traction"]
    mass = preset["mass"]
    length = preset["length"]
    power = preset["power"]
    aux_power = preset["aux_power"]
    accel = preset["accel"]
    decel = preset["decel"]
    st.sidebar.info(f"**{vehicle_choice} specs loaded:**\n"
                    f"- Traction: {traction}\n"
                    f"- Mass: {mass:,} kg\n"
                    f"- Length: {length} m\n"
                    f"- Power: {power} kW\n"
                    f"- Aux Power: {aux_power} kW")

st.sidebar.header("2. Route Settings")
start_name = st.sidebar.selectbox("Start Station", station_names, index=0)
end_name = st.sidebar.selectbox("End Station", station_names, index=len(station_names) - 1)
start_km = station_dict[start_name]
end_km = station_dict[end_name]

st.sidebar.header("3. Stop Policy")
mode = st.sidebar.selectbox("Request Stop Mode", ["random", "all", "none"], index=0)
prob = st.sidebar.slider("Request Probability", 0.0, 1.0, 0.6, 0.05, disabled=(mode != "random"))
dwell = st.sidebar.number_input("Dwell Time (s)", value=30, step=5)

if st.sidebar.button("▶ Run Simulation", type="primary", use_container_width=True):
    if start_km == end_km:
        st.warning("Start and End stations cannot be the same!")
        st.stop()

    with st.spinner("Simulating Train Physics..."):
        try:
            is_fwd = start_km > end_km
            track = TrackProfile("track_profile.xlsx", is_forward_direction=is_fwd)
            sim = TrainSimulator(mass, length, power, aux_power, accel, decel, traction)

            history, stops, stats_curr = sim.run_simulation(track, start_km, end_km, mode, prob, dwell)
            _, _, stats_all = sim.run_simulation(track, start_km, end_km, "all", 1.0, dwell)
            _, _, stats_none = sim.run_simulation(track, start_km, end_km, "none", 0.0, dwell)

            st.subheader("Journey Overview")
            m = int(stats_curr["journey_time_s"] // 60)
            s = int(stats_curr["journey_time_s"] % 60)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Time", f"{m:02d}:{s:02d}")
            col2.metric("Stops Made", len(stops))

            if traction == "DIESEL":
                l_used = stats_curr["net_kwh"] / (10.0 * 0.35)
                l_saved = (stats_all["net_kwh"] - stats_curr["net_kwh"]) / (10.0 * 0.35)
                col3.metric("Fuel Consumed", f"{l_used:.1f} L")
                col4.metric("Fuel Saved", f"{l_saved:.1f} L", help="Compared to stopping at every request stop.")
            else:
                col3.metric("Net Energy", f"{stats_curr['net_kwh']:.1f} kWh")
                col4.metric("Energy Saved", f"{stats_all['net_kwh'] - stats_curr['net_kwh']:.1f} kWh",
                            help="Compared to stopping at every request stop.")

            st.subheader("Performance Telemetry")
            fig = create_plotly_figure(history, stops, track.stations, traction == "ELECTRIC")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Energy Report")
            if traction == "DIESEL":
                base_val, best_val, curr_val = stats_all["net_kwh"] / 3.5, stats_none["net_kwh"] / 3.5, stats_curr[
                    "net_kwh"] / 3.5
                unit = "Liters"
            else:
                base_val, best_val, curr_val = stats_all["net_kwh"], stats_none["net_kwh"], stats_curr["net_kwh"]
                unit = "kWh"

            report_markdown = f"""
            | Metric | {unit} Consumed | Notes |
            | :--- | :--- | :--- |
            | **Baseline (Worst Case)** | `{base_val:.1f}` {unit} | If the train stopped at *every* request stop. |
            | **Optimal (Best Case)** | `{best_val:.1f}` {unit} | If the train skipped *every* request stop. |
            | **This Run ({mode.upper()})** | `{curr_val:.1f}` {unit} | The actual consumption of this simulation. |
            | **Total Saved** | **`{base_val - curr_val:.1f}` {unit}** | Savings achieved in this run compared to the baseline. |
            """
            st.markdown(report_markdown)

            st.subheader("Detailed Stop Log")
            st.write(f"**Requested stops made:** {', '.join(stops) if stops else 'None'}")

            all_rq = [s["name"] for s in track.stations if s["type"] == "R"]
            skipped = [name for name in all_rq if name not in stops]
            st.write(f"**Requested stops skipped:** {', '.join(skipped) if skipped else 'None'}")

        except Exception as e:
            st.error(f"Error reading track_profile.xlsx: {e}")
else:
    st.info("Adjust your parameters in the sidebar and click **Run Simulation** to begin.")
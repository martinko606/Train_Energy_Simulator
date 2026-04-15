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
        "power": 152, "aux_power": 20, "accel": 0.5, "decel": 0.8
    },
    "EDITA+Btax": {
        "traction": "DIESEL", "mass": 42000, "length": 30,
        "power": 152, "aux_power": 25, "accel": 0.4, "decel": 0.8
    },
}

# ==========================================
#  STREAMLIT PAGE CONFIG & STATE
# ==========================================
st.set_page_config(page_title="Railway Energy Simulator", layout="wide")

if "journey_legs" not in st.session_state:
    st.session_state.journey_legs = []


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

        # We can now reduce vertical spacing since names are safely inside the top graph
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

            # Place the text INSIDE the top graph, just slightly above the 0 line
            annotations.append(dict(
                x=x_pos, y=0.03, xref="x", yref="y domain",
                text=f" {station['name'].title()} ",
                showarrow=False, font=dict(size=11, color=color),
                xanchor="center", yanchor="bottom",
                bgcolor="rgba(0,0,0,0.4)"  # Subtle translucent background so graph lines don't obscure text
            ))
        except ValueError:
            pass

    fig.update_layout(
        height=750, margin=dict(l=40, r=40, t=40, b=40), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        # Title is now safely placed on BOTH graphs
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
else:
    preset = PREDEFINED_VEHICLES[vehicle_choice]
    traction = preset["traction"]
    mass = preset["mass"]
    length = preset["length"]
    power = preset["power"]
    aux_power = preset["aux_power"]
    accel = preset["accel"]
    decel = preset["decel"]
    st.sidebar.info(f"**{vehicle_choice} specs loaded.**")

st.sidebar.header("2. Route Settings")

# Lock the start station if we already have legs in the journey!
if st.session_state.journey_legs:
    last_leg_end = st.session_state.journey_legs[-1]["end_name"]
    start_name = st.sidebar.selectbox("Start Station (Locked to previous)", [last_leg_end], disabled=True)
else:
    start_name = st.sidebar.selectbox("Start Station", station_names, index=0)

end_name = st.sidebar.selectbox("End Station", station_names, index=len(station_names) - 1)
start_km = station_dict[start_name]
end_km = station_dict[end_name]

st.sidebar.header("3. Stop Policy")
mode = st.sidebar.selectbox("Request Stop Mode", ["random", "all", "none"], index=0)
prob = st.sidebar.slider("Request Probability", 0.0, 1.0, 0.6, 0.05, disabled=(mode != "random"))
dwell = st.sidebar.number_input("Dwell Time (s)", value=30, step=5)

st.sidebar.header("4. Display Options")
x_axis_choice = st.sidebar.radio("X-Axis Display", ["Time (MM:SS)", "Distance (km)"])

st.sidebar.markdown("---")
col_add, col_reset = st.sidebar.columns(2)

if col_reset.button("🗑️ Reset", use_container_width=True):
    st.session_state.journey_legs = []
    st.rerun()

if col_add.button("▶ Add Leg", type="primary", use_container_width=True):
    if start_km == end_km:
        st.warning("Start and End stations cannot be the same!")
        st.stop()

    with st.spinner("Simulating..."):
        is_fwd = start_km > end_km
        track = TrackProfile("track_profile.xlsx", is_forward_direction=is_fwd)
        sim = TrainSimulator(mass, length, power, aux_power, accel, decel, traction)

        history, stops, stats_curr = sim.run_simulation(track, start_km, end_km, mode, prob, dwell)
        _, _, stats_all = sim.run_simulation(track, start_km, end_km, "all", 1.0, dwell)
        _, _, stats_none = sim.run_simulation(track, start_km, end_km, "none", 0.0, dwell)

        # Save this leg to memory
        st.session_state.journey_legs.append({
            "start_name": start_name, "end_name": end_name,
            "traction": traction, "mode": mode, "prob": prob,
            "history": history, "stops": stops, "stats_curr": stats_curr,
            "stats_all": stats_all, "stats_none": stats_none
        })

# ==========================================
#  MAIN VIEW RENDERER
# ==========================================
if not st.session_state.journey_legs:
    st.info("👈 **Configure your route in the sidebar and click 'Add Leg' to begin your journey.**")
else:
    # --- CUMULATIVE MATH ---
    cum_time_s = sum(leg["stats_curr"]["journey_time_s"] for leg in st.session_state.journey_legs)
    cum_stops = sum(len(leg["stops"]) for leg in st.session_state.journey_legs)

    # We will separate Diesel and Electric metrics purely for the cumulative report
    cum_liters_curr, cum_liters_all, cum_liters_none = 0.0, 0.0, 0.0
    cum_kwh_curr, cum_kwh_all, cum_kwh_none = 0.0, 0.0, 0.0

    for leg in st.session_state.journey_legs:
        if leg["traction"] == "DIESEL":
            cum_liters_curr += leg["stats_curr"]["net_kwh"] / 3.5
            cum_liters_all += leg["stats_all"]["net_kwh"] / 3.5
            cum_liters_none += leg["stats_none"]["net_kwh"] / 3.5
        else:
            cum_kwh_curr += leg["stats_curr"]["net_kwh"]
            cum_kwh_all += leg["stats_all"]["net_kwh"]
            cum_kwh_none += leg["stats_none"]["net_kwh"]

    # --- TOP DASHBOARD ---
    st.subheader(f"🏁 Cumulative Journey Summary ({len(st.session_state.journey_legs)} Legs)")
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
    st.subheader("🗺️ Route Breakdown")

    for i, leg in enumerate(st.session_state.journey_legs):
        # Default to expanding only the most recently added leg
        is_expanded = (i == len(st.session_state.journey_legs) - 1)

        with st.expander(
                f"Leg {i + 1}: {leg['start_name']} ➔ {leg['end_name']} ({leg['traction']} | Mode: {leg['mode'].upper()})",
                expanded=is_expanded):

            fig = create_plotly_figure(leg["history"], leg["stops"], TrackProfile("track_profile.xlsx", True).stations,
                                       leg["traction"] == "ELECTRIC", x_axis_choice)
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

            # Local Leg Energy Report
            if leg["traction"] == "DIESEL":
                base_val = leg["stats_all"]["net_kwh"] / 3.5
                curr_val = leg["stats_curr"]["net_kwh"] / 3.5
                unit = "Liters"
            else:
                base_val = leg["stats_all"]["net_kwh"]
                curr_val = leg["stats_curr"]["net_kwh"]
                unit = "kWh"

            st.markdown(
                f"**Leg Energy Metrics:** `{curr_val:.1f}` {unit} used | `{base_val - curr_val:.1f}` {unit} saved by skipping stops.")
            st.caption(f"**Stops Made:** {', '.join(leg['stops']) if leg['stops'] else 'None'}")
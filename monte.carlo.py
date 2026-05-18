import math
import glob
import pandas as pd
import random
import numpy as np
import plotly.express as px
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

    def run_simulation(self, track, start_km, end_km, stop_mode, stop_prob, dwell_time):
        total_dist_m = abs(start_km - end_km) * 1000
        direction = -1 if start_km > end_km else 1
        events, stops_km, stops_names = self._build_events(track, start_km, end_km, stop_mode, stop_prob, direction)

        dt, current_v, distance_covered, total_energy_j, total_regen_j, journey_time_s = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
        g_accel = 9.8186

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
                total_energy_j += self.aux_power_w * dwell_time
                journey_time_s += dwell_time

                events = [e for e in events if not (e["type"] == "stop" and abs(e["km"] - current_km) <= 0.05)]
                stops_km = [s for s in stops_km if abs(s - current_km) > 0.05]

        return None, stops_names, {
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


# ==========================================
#  MAIN DASHBOARD
# ==========================================
st.title("🎲 Monte Carlo Fleet Simulator")
st.markdown("Dedicated tool for calculating stochastic expected values for regional and mainline railway shifts.")

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

# --- RUN BUTTON ---
if st.sidebar.button("🎲 Run Monte Carlo Analysis", type="primary", use_container_width=True):
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


            # Helper to calculate appropriate units
            def get_unit(stats):
                return stats["net_kwh"] / (diesel_density * efficiency) if traction == "DIESEL" else stats["net_kwh"]


            # Shift Logic Variables
            n_fwd = num_cycles
            n_rev = num_cycles if trip_pattern == "Round Trip (A ➔ B ➔ A)" else 0

            # Deterministic Baselines
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

            # Scale baselines by the number of cycles
            base_worst_unit = (get_unit(w_stats_1) * n_fwd) + (get_unit(w_stats_2) * n_rev)
            base_best_unit = (get_unit(b_stats_1) * n_fwd) + (get_unit(b_stats_2) * n_rev)

            base_worst_time = (w_stats_1["journey_time_s"] * n_fwd) + (w_stats_2["journey_time_s"] * n_rev)
            base_best_time = (b_stats_1["journey_time_s"] * n_fwd) + (b_stats_2["journey_time_s"] * n_rev)

            base_worst_stops = (len(w_stops_1) * n_fwd) + (len(w_stops_2) * n_rev)
            base_best_stops = (len(b_stops_1) * n_fwd) + (len(b_stops_2) * n_rev)

            # Stochastic Loops
            results = []
            for p in [0.2, 0.4, 0.6, 0.8]:
                t_sum, e_sum, s_sum = 0.0, 0.0, 0.0
                for _ in range(int(mc_runs)):

                    # Execute Forward Legs
                    for _f in range(n_fwd):
                        _, r_stops, r_stats = sim.run_simulation(track_1, station_dict[mc_start], station_dict[mc_end],
                                                                 "random", p, mc_dwell)
                        t_sum += r_stats["journey_time_s"]
                        e_sum += get_unit(r_stats)
                        s_sum += len(r_stops)

                    # Execute Return Legs
                    for _r in range(n_rev):
                        _, r_stops, r_stats = sim.run_simulation(track_2, station_dict[mc_end], station_dict[mc_start],
                                                                 "random", p, mc_dwell)
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

            df = pd.DataFrame(results)
            st.session_state.mc_results = {
                "df": df, "start": mc_start, "end": mc_end, "runs": mc_runs,
                "unit": "Liters" if traction == "DIESEL" else "kWh",
                "total_legs": n_fwd + n_rev,
                "cycles": num_cycles,
                "pattern": trip_pattern
            }
        except Exception as e:
            st.error(f"Error during Monte Carlo: {e}")

# ==========================================
#  MAIN VIEW RENDERER
# ==========================================
if st.session_state.mc_results is None:
    st.info("👈 **Configure your Shift and Monte Carlo parameters in the sidebar, then click 'Run'.**")
else:
    mc = st.session_state.mc_results
    df = mc["df"]
    unit = mc["unit"]

    st.subheader(f"🎲 Monte Carlo Expected Values: {mc['start']} ➔ {mc['end']}")
    st.markdown(f"**Configuration:** {mc['pattern']} | **Cycles:** {mc['cycles']} | **Total Legs:** {mc['total_legs']}")
    st.caption(
        f"Based on **{mc['runs']}** iterations per probability tier. Data loaded from **{selected_track_file}**.")

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
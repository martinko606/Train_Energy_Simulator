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
        self.mass_kg, self.length_m = mass_kg, length_m
        self.eff_mass = self.mass_kg * 1.08
        self.A, self.B, self.C = 1500, 30, 4
        self.traction_efficiency = efficiency if traction_type == "ELECTRIC" else 0.85
        self.regen_efficiency = 0.75 if traction_type == "ELECTRIC" else 0.0
        self.aux_power_w = aux_power_kw * 1000
        self.max_power_w = max_power_kw * 1000
        self.max_accel, self.max_decel = max_accel, max_decel

    def get_resistance(self, v_m_s: float) -> float:
        return self.A + (self.B * v_m_s) + (self.C * v_m_s ** 2)

    def _build_events(self, track, start_km, end_km, stop_mode, stop_prob, direction):
        km_min, km_max = min(start_km, end_km), max(start_km, end_km)
        stops_km, stops_names, events = [], [], []
        for station in track.stations:
            km = station["km"]
            if not (km_min <= km <= km_max) or abs(km - start_km) < 0.01: continue
            will_stop = station["type"] == "X" or (stop_mode == "all") or (
                        stop_mode == "random" and random.random() <= stop_prob)
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
        total_dist_m, direction = abs(start_km - end_km) * 1000, -1 if start_km > end_km else 1
        events, stops_km, stops_names = self._build_events(track, start_km, end_km, stop_mode, stop_prob, direction)
        dt, current_v, dist_c, e_j, r_j, time_s = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
        history = {"time_s": [], "km": [], "cum_dist_km": [], "v_actual": [], "v_limit": [], "energy_kwh": [],
                   "regen_kwh": []} if record_history else None

        while dist_c < total_dist_m:
            cur_km = start_km + (dist_c / 1000.0) * direction
            lim, slope = track.get_effective_limit_and_grad(cur_km, cur_km - (self.length_m / 1000.0) * direction)

            f_grad = self.mass_kg * 9.8186 * slope
            eff_dec = max(0.05, self.max_decel + (9.8186 * slope))
            max_safe = lim

            for ev in events:
                tol = 0.05 if ev["type"] == "stop" else 1e-4
                if (ev["km"] <= cur_km + tol) if direction == -1 else (ev["km"] >= cur_km - tol):
                    d_ev = 0.0 if (ev["type"] == "stop" and (
                        (cur_km < ev["km"]) if direction == -1 else (cur_km > ev["km"]))) else abs(
                        cur_km - ev["km"]) * 1000
                    if ev["target_v"] < current_v: max_safe = min(max_safe, math.sqrt(
                        max(0.0, ev["target_v"] ** 2 + 2 * eff_dec * d_ev)))

            if current_v > max_safe + 1e-4:
                acc = -max(self.max_decel, (current_v - max_safe) / dt)
                r_j += (self.eff_mass * abs(acc)) * current_v * self.regen_efficiency * dt
                current_v = max(0.0, current_v + acc * dt)
            elif current_v < min(lim, max_safe) - 1e-4:
                f_res = self.get_resistance(current_v)
                f_act = max(0.0, min(f_res + (self.eff_mass * self.max_accel) + f_grad,
                                     self.max_power_w / max(current_v, 0.5)))
                current_v = min(current_v + ((f_act - f_res - f_grad) / self.eff_mass) * dt, lim, max_safe)
                e_j += (f_act / self.traction_efficiency + self.aux_power_w) * dt
            else:
                f_res = self.get_resistance(current_v)
                current_v += ((max(0.0, min(f_res + f_grad, self.max_power_w / max(current_v,
                                                                                   0.5))) - f_res - f_grad) / self.eff_mass) * dt
                e_j += (self.aux_power_w) * dt

            dist_c += current_v * dt
            time_s += dt
            if current_v < 0.5 and any(abs(cur_km - s) <= 0.05 for s in stops_km):
                current_v, time_s, e_j = 0.0, time_s + dwell_time, e_j + (self.aux_power_w * dwell_time)
                events = [e for e in events if not (e["type"] == "stop" and abs(e["km"] - cur_km) <= 0.05)]
                stops_km = [s for s in stops_km if abs(s - cur_km) > 0.05]
        return history, stops_names, {"net_kwh": (e_j - r_j) / 3_600_000.0, "journey_time_s": time_s}


# ==========================================
#  MAIN DASHBOARD
# ==========================================
st.title("🎲 Monte Carlo Fleet Simulator")

excel_files = [f for f in glob.glob("*.xlsx") if not f.startswith("~$")]
if not excel_files: st.sidebar.error("No Excel files found."); st.stop()
selected_track = st.sidebar.selectbox("Select Track Profile", excel_files)
mandatory_stations = load_mandatory_stations(selected_track)
station_dict = {s[0]: s[1] for s in mandatory_stations}
station_names = [s[0] for s in mandatory_stations]

# [Parameters Setup - Simplified for brevity]
vehicle_choice = st.sidebar.selectbox("Vehicle", ["Custom"] + list(PREDEFINED_VEHICLES.keys()))
preset = PREDEFINED_VEHICLES.get(vehicle_choice,
                                 {"traction": "DIESEL", "mass": 100000, "length": 40, "power": 600, "aux_power": 40,
                                  "accel": 0.6, "decel": 0.8, "efficiency": 0.35})
mc_start, mc_end = st.sidebar.selectbox("Terminal A", station_names, index=0), st.sidebar.selectbox("Terminal B",
                                                                                                    station_names,
                                                                                                    index=len(
                                                                                                        station_names) - 1)
trip_pattern = st.sidebar.radio("Trip Pattern", ["Single Direction (A ➔ B)", "Round Trip (A ➔ B ➔ A)"])
num_cycles = st.sidebar.number_input("Cycles", 1, 20, 1)
mc_runs = st.sidebar.number_input("N (Runs)", 10, 1000, 100)
mc_dwell = st.sidebar.number_input("Dwell Time (s)", 30, 300, 30)

if st.sidebar.button("Run MC", type="primary"):
    track_1 = TrackProfile(selected_track, station_dict[mc_start] > station_dict[mc_end])
    track_2 = TrackProfile(selected_track, station_dict[mc_end] > station_dict[mc_start])
    sim = TrainSimulator(preset["mass"], preset["length"], preset["power"], preset["aux_power"], preset["accel"],
                         preset["decel"], preset["traction"], preset["efficiency"])

    # Baseline calc
    _, w_stops_1, _ = sim.run_simulation(track_1, station_dict[mc_start], station_dict[mc_end], "all", 1.0, mc_dwell)
    _, b_stops_1, _ = sim.run_simulation(track_1, station_dict[mc_start], station_dict[mc_end], "none", 0.0, mc_dwell)
    n_rev = num_cycles if trip_pattern == "Round Trip (A ➔ B ➔ A)" else 0

    mand_per_leg = len(b_stops_1)
    req_per_leg = len(w_stops_1) - len(b_stops_1)

    results = []
    for p in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
        t_s, e_s, s_s = 0.0, 0.0, 0.0
        for _ in range(mc_runs):
            for _f in range(num_cycles):
                _, r_s, r_st = sim.run_simulation(track_1, station_dict[mc_start], station_dict[mc_end], "random", p,
                                                  mc_dwell)
                t_s, e_s, s_s = t_s + r_st["journey_time_s"], e_s + r_st["net_kwh"], s_s + len(r_s)
            for _r in range(n_rev):
                _, r_s, r_st = sim.run_simulation(track_2, station_dict[mc_end], station_dict[mc_start], "random", p,
                                                  mc_dwell)
                t_s, e_s, s_s = t_s + r_st["journey_time_s"], e_s + r_st["net_kwh"], s_s + len(r_s)

        results.append({
            "Probability": f"{int(p * 100)}%",
            "Avg Stops Made": round(s_s / mc_runs, 1),
            "Req. Stops Made/Leg": round(
                ((s_s / mc_runs) - (mand_per_leg * (num_cycles + n_rev))) / (num_cycles + n_rev), 2),
            "Avg Time": f"{int(t_s / mc_runs // 60)}m {int(t_s / mc_runs % 60)}s",
            "Energy (kWh)": round(e_s / mc_runs, 2)
        })

    st.session_state.mc_results = {"df": pd.DataFrame(results), "mand_per_leg": mand_per_leg,
                                   "req_per_leg": req_per_leg}

if st.session_state.mc_results:
    mc = st.session_state.mc_results
    st.subheader(f"Results (Mandatory: {mc['mand_per_leg']} / Request: {mc['req_per_leg']} stops per leg)")
    st.dataframe(mc["df"], use_container_width=True)
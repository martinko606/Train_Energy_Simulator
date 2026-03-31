"""
Train Energy Simulation
=======================
Outputs (all saved inside ./results/):
  • speed_{direction}_{mode}_p{prob}.png      – speed-profile static chart
  • energy_{direction}_{mode}_p{prob}.png     – energy static chart
  • interactive_{direction}_{mode}_p{prob}.html – zoomable / pannable Plotly chart
  • report_{direction}_{mode}_p{prob}.txt     – summary text report
"""

import math
import os
import pandas as pd
import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# --- SIMULATION SETTINGS & FLAGS ---
# ==========================================
EXCEL_FILE = "track_profile.xlsx"
RESULTS_DIR = "results"

# --- Route & Timetable Settings ---
START_KM = 22.376  # Dolní Bousov
END_KM = 0.00  # Kopidlno
IS_FORWARD = True  # True if km values are DECREASING (22.376 → 0)

STOP_PROBABILITY = 0.6
REQUEST_STOP_MODE = "random"  # "all" | "none" | "random"
DWELL_TIME_S = 30  # Seconds spent waiting at the platform

# --- Train Physical Parameters ---
TRAIN_TRACTION_TYPE = "DIESEL"  # Options: "ELECTRIC" or "DIESEL"
TRAIN_MASS_KG = 35_000
TRAIN_LENGTH_M = 28  # Length of train in meters
TRAIN_MAX_POWER_KW = 155  # Maximum traction power in Kilowatts
TRAIN_AUX_POWER_KW = 10  # Auxiliary power
MAX_ACCEL = 0.6  # Theoretical max acceleration (m/s^2)
MAX_DECEL = 0.8  # Braking deceleration (m/s^2)

# --- Electrical & Diesel Efficiencies ---
ELEC_EFFICIENCY = 0.85  # From overhead wire to wheel
REGEN_EFFICIENCY = 0.75  # How much braking energy returns to the grid
DIESEL_EFFICIENCY = 0.35  # Diesel engine thermal efficiency (~35%)
DIESEL_KWH_PER_L = 10.0  # Approx energy density of diesel fuel (kWh/L)


# ==========================================
#  HELPERS
# ==========================================
def build_filename(prefix: str, mode: str, prob: float, direction: str, ext: str) -> str:
    prob_tag = f"p{int(prob * 100)}"
    return f"{prefix}_{direction}_{mode}_{prob_tag}.{ext.lstrip('.')}"


def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


# ==========================================
#  TRACK PROFILE
# ==========================================
class TrackProfile:
    def __init__(self, excel_filepath: str, is_forward_direction: bool):
        self.is_forward = is_forward_direction
        self.df_raw = self._load_from_excel(excel_filepath)
        self.segments = self._build_segments()
        self.stations = self._extract_stations()

    def _load_from_excel(self, filepath: str) -> pd.DataFrame:
        print(f"Loading track data from: {filepath} …")
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
        span_min = min(front_km, rear_km)
        span_max = max(front_km, rear_km)
        limits = []
        current_grad = 0.0
        eps = 1e-6
        for seg in self.segments:
            if span_min <= seg["km_high"] + eps and span_max >= seg["km_low"] - eps:
                limits.append(seg["v_limit"])
            if seg["km_low"] - eps <= front_km <= seg["km_high"] + eps:
                current_grad = seg["grad"]
        eff_limit = min(limits) if limits else 0.0
        return eff_limit, current_grad


# ==========================================
#  SIMULATOR
# ==========================================
class TrainSimulator:
    def __init__(self):
        self.mass_kg = TRAIN_MASS_KG
        self.eff_mass = self.mass_kg * 1.08
        self.A, self.B, self.C = 1500, 30, 4
        self.traction_efficiency = ELEC_EFFICIENCY
        self.regen_efficiency = REGEN_EFFICIENCY if TRAIN_TRACTION_TYPE == "ELECTRIC" else 0.0
        self.aux_power_w = TRAIN_AUX_POWER_KW * 1000
        self.max_power_w = TRAIN_MAX_POWER_KW * 1000
        self.max_accel = MAX_ACCEL
        self.max_decel = MAX_DECEL
        self.train_length_m = TRAIN_LENGTH_M

    def get_resistance(self, v_m_s: float) -> float:
        return self.A + (self.B * v_m_s) + (self.C * v_m_s ** 2)

    def _build_events(self, track, start_km, end_km, stop_mode, stop_prob, direction):
        km_min, km_max = min(start_km, end_km), max(start_km, end_km)
        stops_km, stops_names, events = [], [], []

        for station in track.stations:
            km = station["km"]
            if not (km_min <= km <= km_max): continue
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

    def run_simulation(self, track, start_km, end_km, stop_mode, stop_prob):
        total_dist_m = abs(start_km - end_km) * 1000
        direction = -1 if start_km > end_km else 1

        events, stops_km, stops_names = self._build_events(track, start_km, end_km, stop_mode, stop_prob, direction)

        dt = 1.0
        current_v = 0.0
        distance_covered = 0.0
        total_energy_j = 0.0
        total_regen_j = 0.0
        journey_time_s = 0.0

        history = {"time_s": [], "km": [], "v_actual": [], "v_limit": [], "energy_kwh": [], "regen_kwh": []}

        while distance_covered < total_dist_m:
            current_km = start_km + (distance_covered / 1000.0) * direction
            rear_km = current_km - (self.train_length_m / 1000.0) * direction
            current_limit_m_s, slope = track.get_effective_limit_and_grad(current_km, rear_km)

            f_gradient = self.mass_kg * 9.81 * slope
            max_safe_v = current_limit_m_s

            for event in events:
                tolerance = 0.05 if event["type"] == "stop" else 1e-4
                if direction == -1:
                    is_ahead = event["km"] <= current_km + tolerance
                    overshot = current_km < event["km"]
                else:
                    is_ahead = event["km"] >= current_km - tolerance
                    overshot = current_km > event["km"]

                if not is_ahead: continue
                dist_to_event = abs(current_km - event["km"]) * 1000
                if event["type"] == "stop" and overshot: dist_to_event = 0.0

                if event["target_v"] < current_v:
                    safe_v = math.sqrt(max(0.0, event["target_v"] ** 2 + 2 * self.max_decel * dist_to_event))
                    max_safe_v = min(max_safe_v, safe_v)

            mech_power = 0.0
            regen_power = 0.0

            if current_v > max_safe_v + 1e-4:
                # BRAKING
                decel = min(self.max_decel, (current_v - max_safe_v) / dt)
                regen_power = (self.eff_mass * decel) * current_v * self.regen_efficiency
                current_v -= decel * dt
                if current_v < 0: current_v = 0.0

            elif current_v < min(current_limit_m_s, max_safe_v) - 1e-4:
                # ACCELERATING (Power-Limited Physics)
                f_resist = self.get_resistance(current_v)
                f_accel_desired = self.eff_mass * self.max_accel
                total_desired_force = f_resist + f_accel_desired + f_gradient

                # Available force drops at high speeds: F = P / v
                # (Use max(0.5) to avoid dividing by zero at a standstill)
                max_available_force = self.max_power_w / max(current_v, 0.5)

                actual_force = min(total_desired_force, max_available_force)
                net_force = actual_force - f_resist - f_gradient

                actual_accel = net_force / self.eff_mass
                current_v += actual_accel * dt
                current_v = min(current_v, current_limit_m_s, max_safe_v)
                current_v = max(0.0, current_v)  # Can't roll backwards in this simple model

                mech_power = max(0.0, actual_force * current_v)
            else:
                # CRUISING
                f_resist = self.get_resistance(current_v)
                actual_force = f_resist + f_gradient
                mech_power = max(0.0, actual_force * current_v)

                # If climbing a hill requires more power than the engine has, train slows down!
                if mech_power > self.max_power_w:
                    actual_force = self.max_power_w / max(current_v, 0.5)
                    net_force = actual_force - f_resist - f_gradient
                    actual_accel = net_force / self.eff_mass
                    current_v += actual_accel * dt
                    mech_power = self.max_power_w

            history["time_s"].append(journey_time_s)
            history["km"].append(current_km)
            history["v_actual"].append(current_v * 3.6)
            history["v_limit"].append(current_limit_m_s * 3.6)
            history["energy_kwh"].append(total_energy_j / 3_600_000.0)
            history["regen_kwh"].append(total_regen_j / 3_600_000.0)

            # Diesel trains burn fuel even to generate mechanical power, electric draws from grid
            elec_power = mech_power / self.traction_efficiency
            total_energy_j += (elec_power + self.aux_power_w) * dt
            total_regen_j += regen_power * dt
            distance_covered += current_v * dt
            journey_time_s += dt

            # DWELL TRIGGER
            if current_v < 0.5 and any(abs(current_km - s) <= 0.05 for s in stops_km):
                current_v = 0.0
                history["time_s"].append(journey_time_s)
                history["km"].append(current_km)
                history["v_actual"].append(0.0)
                history["v_limit"].append(current_limit_m_s * 3.6)
                history["energy_kwh"].append(total_energy_j / 3_600_000.0)
                history["regen_kwh"].append(total_regen_j / 3_600_000.0)

                total_energy_j += self.aux_power_w * DWELL_TIME_S
                journey_time_s += DWELL_TIME_S

                history["time_s"].append(journey_time_s)
                history["km"].append(current_km)
                history["v_actual"].append(0.0)
                history["v_limit"].append(current_limit_m_s * 3.6)
                history["energy_kwh"].append(total_energy_j / 3_600_000.0)
                history["regen_kwh"].append(total_regen_j / 3_600_000.0)

                events = [e for e in events if not (e["type"] == "stop" and abs(e["km"] - current_km) <= 0.05)]
                stops_km = [s for s in stops_km if abs(s - current_km) > 0.05]

        stats = {
            "total_kwh": total_energy_j / 3_600_000.0,
            "regen_kwh": total_regen_j / 3_600_000.0,
            "net_kwh": (total_energy_j - total_regen_j) / 3_600_000.0,
            "journey_time_s": journey_time_s,
        }
        return history, stops_names, stats


# ==========================================
#  PLOTTING LOGIC
# ==========================================
_BG_FIG, _BG_AX, _GRID, _TEXT = "#0d1117", "#161b22", "#21262d", "#e6edf3"
_COL_LIMIT, _COL_ACTUAL = "#f85149", "#3fb950"
_COL_GROSS, _COL_REGEN, _COL_NET = "#d29922", "#58a6ff", "#bc8cff"
_COL_STOP_X, _COL_STOP_MADE, _COL_STOP_SKIP = "#ffffff", "#79c0ff", "#484f58"


def _station_color(station: dict, stops_set: set) -> str:
    if station["type"] == "X": return _COL_STOP_X
    return _COL_STOP_MADE if station["name"] in stops_set else _COL_STOP_SKIP


def _apply_dark_style(ax):
    ax.set_facecolor(_BG_AX)
    ax.tick_params(colors=_TEXT, labelsize=8)
    ax.yaxis.label.set_color(_TEXT)
    ax.xaxis.label.set_color(_TEXT)
    for spine in ax.spines.values(): spine.set_edgecolor("#30363d")
    ax.grid(True, color=_GRID, linewidth=0.5)


def _save_fig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_station_vlines(ax, history, stations, stops_set):
    trans = ax.get_xaxis_transform()
    for station in stations:
        skm = station["km"]
        color = _station_color(station, stops_set)
        try:
            idx = (np.abs(np.array(history["km"]) - skm)).argmin()
            time_at_station = history["time_s"][idx]
            ax.axvline(x=time_at_station, color=color, linewidth=0.9, linestyle="--", alpha=0.65, zorder=2)
            ax.text(time_at_station, -0.09, station["name"], transform=trans, rotation=0, va="top", ha="center",
                    fontsize=8, color=color, fontfamily="monospace", clip_on=False, zorder=5)
        except ValueError:
            pass


def time_formatter(x, pos): return format_time(x)


def plot_speed(history, stops_names, all_stations, stats, title, filepath):
    time_arr = np.array(history["time_s"])
    stops_set = set(stops_names)
    fig, ax = plt.subplots(figsize=(18, 5))
    fig.patch.set_facecolor(_BG_FIG)
    _apply_dark_style(ax)
    fig.suptitle(title, color=_TEXT, fontsize=12, fontweight="bold", y=1.01)
    ax.plot(time_arr, history["v_limit"], color=_COL_LIMIT, linestyle="--", linewidth=1.5,
            label="Effective Speed Limit (km/h)", zorder=3)
    ax.plot(time_arr, history["v_actual"], color=_COL_ACTUAL, linewidth=2, label="Actual Speed (km/h)", zorder=4)
    ax.fill_between(time_arr, history["v_actual"], color=_COL_ACTUAL, alpha=0.08)
    ax.set_ylabel("Speed  (km/h)", color=_TEXT, fontsize=10)
    ax.set_xlabel("Time (MM:SS)", color=_TEXT, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))
    _draw_station_vlines(ax, history, all_stations, stops_set)
    patches = [mpatches.Patch(color=_COL_STOP_X, label="Mandatory stop (X)"),
               mpatches.Patch(color=_COL_STOP_MADE, label="Request stop – made"),
               mpatches.Patch(color=_COL_STOP_SKIP, label="Request stop – skipped")]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + patches, facecolor=_BG_FIG, edgecolor="#30363d", labelcolor=_TEXT, fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    _save_fig(fig, filepath)


def plot_energy(history, stops_names, all_stations, stats, title, filepath):
    time_arr = np.array(history["time_s"])
    stops_set = set(stops_names)
    net = [g - r for g, r in zip(history["energy_kwh"], history["regen_kwh"])]
    fig, ax = plt.subplots(figsize=(18, 5))
    fig.patch.set_facecolor(_BG_FIG)
    _apply_dark_style(ax)
    fig.suptitle(title, color=_TEXT, fontsize=12, fontweight="bold", y=1.01)
    ax.plot(time_arr, history["energy_kwh"], color=_COL_GROSS, linewidth=2, label="Gross Energy Draw (kWh)", zorder=4)
    ax.fill_between(time_arr, history["energy_kwh"], color=_COL_GROSS, alpha=0.08)
    if TRAIN_TRACTION_TYPE == "ELECTRIC":
        ax.plot(time_arr, history["regen_kwh"], color=_COL_REGEN, linewidth=1.5, linestyle=":",
                label="Regen Recovered (kWh)", zorder=3)
        ax.fill_between(time_arr, history["regen_kwh"], color=_COL_REGEN, alpha=0.07)
    ax.plot(time_arr, net, color=_COL_NET, linewidth=2, linestyle="-.", label="Net Energy (kWh)", zorder=5)
    ax.set_ylabel("Energy  (kWh)", color=_TEXT, fontsize=10)
    ax.set_xlabel("Time (MM:SS)", color=_TEXT, fontsize=10)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))
    _draw_station_vlines(ax, history, all_stations, stops_set)
    patches = [mpatches.Patch(color=_COL_STOP_X, label="Mandatory stop (X)"),
               mpatches.Patch(color=_COL_STOP_MADE, label="Request stop – made"),
               mpatches.Patch(color=_COL_STOP_SKIP, label="Request stop – skipped")]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + patches, facecolor=_BG_FIG, edgecolor="#30363d", labelcolor=_TEXT, fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    _save_fig(fig, filepath)


def plot_interactive(history, stops_names, all_stations, stats, title: str, filepath: str):
    base_time = pd.to_datetime("1970-01-01")
    time_dt_arr = [base_time + pd.to_timedelta(s, unit='s') for s in history["time_s"]]
    stops_set = set(stops_names)
    net = [g - r for g, r in zip(history["energy_kwh"], history["regen_kwh"])]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12)
    fig.add_trace(go.Scatter(x=time_dt_arr, y=history["v_limit"], name="Effective Speed Limit",
                             line=dict(color=_COL_LIMIT, dash="dash", width=1.5)), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=time_dt_arr, y=history["v_actual"], name="Actual Speed", line=dict(color=_COL_ACTUAL, width=2),
                   fill="tozeroy", fillcolor="rgba(63,185,80,0.07)"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=time_dt_arr, y=history["energy_kwh"], name="Gross Energy", line=dict(color=_COL_GROSS, width=2),
                   fill="tozeroy", fillcolor="rgba(210,153,34,0.07)"), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=time_dt_arr, y=net, name="Net Energy", line=dict(color=_COL_NET, width=2, dash="dashdot")), row=2,
        col=1)

    if TRAIN_TRACTION_TYPE == "ELECTRIC":
        fig.add_trace(go.Scatter(x=time_dt_arr, y=history["regen_kwh"], name="Regen Recovered",
                                 line=dict(color=_COL_REGEN, width=1.5, dash="dot"), fill="tozeroy",
                                 fillcolor="rgba(88,166,255,0.06)"), row=2, col=1)

    shapes, annotations = [], []
    v_max = max(max(history["v_limit"]), max(history["v_actual"])) * 1.05
    e_max = max(history["energy_kwh"]) * 1.05
    for station in all_stations:
        skm = station["km"]
        color = _station_color(station, stops_set)
        try:
            idx = (np.abs(np.array(history["km"]) - skm)).argmin()
            time_at_station_dt = time_dt_arr[idx]
            shapes.append(
                dict(type="line", xref="x", yref="y", x0=time_at_station_dt, x1=time_at_station_dt, y0=0, y1=v_max,
                     line=dict(color=color, width=1, dash="dot"), layer="below"))
            shapes.append(
                dict(type="line", xref="x", yref="y2", x0=time_at_station_dt, x1=time_at_station_dt, y0=0, y1=e_max,
                     line=dict(color=color, width=1, dash="dot"), layer="below"))
            annotations.append(
                dict(x=time_at_station_dt, y=-0.06, xref="x", yref="paper", text=station["name"], textangle=0,
                     showarrow=False, font=dict(size=10, color=color, family="Courier New, monospace"),
                     xanchor="center", yanchor="top", bgcolor="rgba(13,17,23,0)"))
        except ValueError:
            pass

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=14, color="#e6edf3"), x=0.01),
        paper_bgcolor=_BG_FIG, plot_bgcolor=_BG_AX, font=dict(color="#e6edf3", family="Courier New, monospace"),
        legend=dict(bgcolor=_BG_AX, bordercolor="#30363d", borderwidth=1, orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        shapes=shapes, annotations=annotations, hovermode="x unified", dragmode="pan", height=760
    )
    plot_div = fig.to_html(full_html=False, include_plotlyjs="cdn")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"<html><body style='background:{_BG_FIG}'>{plot_div}</body></html>")


# ==========================================
#  TEXT REPORT GENERATOR (UPDATED FOR DIESEL)
# ==========================================
def calc_liters(kwh):
    # Liters = Energy / (Energy Density * Engine Thermal Efficiency)
    return kwh / (DIESEL_KWH_PER_L * DIESEL_EFFICIENCY)


def generate_text_report(stats, stats_all, stats_none, stops_made, all_stations, filepath):
    import textwrap
    all_station_names = [s["name"] for s in all_stations if s["type"] in ["X", "R"]]
    skipped_stops = [name for name in all_station_names if name not in stops_made]

    base_kwh = stats_all['net_kwh']
    best_kwh = stats_none['net_kwh']
    curr_kwh = stats['net_kwh']
    max_savings_kwh = base_kwh - best_kwh
    run_savings_kwh = base_kwh - curr_kwh

    report = []
    report.append("=" * 66)
    report.append(f"{'TRAIN ENERGY SIMULATION REPORT':^66}")
    report.append("=" * 66)
    report.append(f"Traction Type:   {TRAIN_TRACTION_TYPE} ({TRAIN_MAX_POWER_KW} kW)")
    report.append(f"Route:           {START_KM} km -> {END_KM} km")
    report.append(f"Mode:            {REQUEST_STOP_MODE.upper()} (Probability: {int(STOP_PROBABILITY * 100)}%)")

    report.append("-" * 66)
    report.append(f"{'ENERGY SAVINGS ANALYSIS':^66}")
    report.append("-" * 66)

    if TRAIN_TRACTION_TYPE == "DIESEL":
        base_l, best_l, curr_l = calc_liters(base_kwh), calc_liters(best_kwh), calc_liters(curr_kwh)
        max_l, run_l = calc_liters(max_savings_kwh), calc_liters(run_savings_kwh)

        report.append(f"Baseline (NO request stops skipped):           {base_l:6.2f} Liters")
        report.append(f"Optimal  (ALL request stops skipped):          {best_l:6.2f} Liters")
        report.append(f"Max Potential Fuel Saving:                     {max_l:6.2f} Liters")
        report.append("")
        report.append(f"This Run ({REQUEST_STOP_MODE.upper()} mode):                        {curr_l:6.2f} Liters")
        report.append(f"Fuel Saved in This Run:                        {run_l:6.2f} Liters")
    else:
        report.append(f"Baseline (NO request stops skipped):           {base_kwh:6.2f} kWh")
        report.append(f"Optimal  (ALL request stops skipped):          {best_kwh:6.2f} kWh")
        report.append(f"Max Potential Energy Saving:                   {max_savings_kwh:6.2f} kWh")
        report.append("")
        report.append(f"This Run ({REQUEST_STOP_MODE.upper()} mode):                        {curr_kwh:6.2f} kWh")
        report.append(f"Energy Saved in This Run:                      {run_savings_kwh:6.2f} kWh")

    report.append("-" * 66)
    report.append(f"{'JOURNEY METRICS (THIS RUN)':^66}")
    report.append("-" * 66)
    report.append(f"Total Time:      {format_time(stats['journey_time_s'])}")
    if TRAIN_TRACTION_TYPE == "DIESEL":
        report.append(f"Total Fuel Consumed:  {calc_liters(curr_kwh):.2f} Liters")
    else:
        report.append(f"Gross Energy:    {stats['total_kwh']:.2f} kWh")
        report.append(f"Regen Energy:   -{stats['regen_kwh']:.2f} kWh")
        report.append(f"Net Energy:      {curr_kwh:.2f} kWh")

    report.append("-" * 66)
    report.append(f"{'STATION SUMMARY':^66}")
    report.append("-" * 66)

    stops_str = ", ".join(stops_made) if stops_made else "None"
    skips_str = ", ".join(skipped_stops) if skipped_stops else "None"

    report.append("Stops Made:")
    for line in textwrap.wrap(stops_str, width=62): report.append(f"    {line}")
    report.append("\nStops Skipped:")
    for line in textwrap.wrap(skips_str, width=62): report.append(f"    {line}")
    report.append("=" * 66)

    report_text = "\n".join(report)
    print(f"\n{report_text}\n")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)


# ==========================================
#  ENTRY POINT
# ==========================================
if __name__ == "__main__":
    track = TrackProfile(EXCEL_FILE, is_forward_direction=IS_FORWARD)
    sim = TrainSimulator()
    direction_tag = "fwd" if IS_FORWARD else "rev"

    print(f"\nRunning Primary Simulation ({REQUEST_STOP_MODE.upper()} mode)...")
    history, stops, stats = sim.run_simulation(track, START_KM, END_KM, stop_mode=REQUEST_STOP_MODE,
                                               stop_prob=STOP_PROBABILITY)

    print("Calculating baselines for energy savings report...")
    _, _, stats_all = sim.run_simulation(track, START_KM, END_KM, stop_mode="all", stop_prob=1.0)
    _, _, stats_none = sim.run_simulation(track, START_KM, END_KM, stop_mode="none", stop_prob=0.0)

    base_title = f"{START_KM} km → {END_KM} km  |  {REQUEST_STOP_MODE.upper()}  p={int(STOP_PROBABILITY * 100)}%"
    speed_path = os.path.join(RESULTS_DIR,
                              build_filename("speed", REQUEST_STOP_MODE, STOP_PROBABILITY, direction_tag, "png"))
    energy_path = os.path.join(RESULTS_DIR,
                               build_filename("energy", REQUEST_STOP_MODE, STOP_PROBABILITY, direction_tag, "png"))
    html_path = os.path.join(RESULTS_DIR,
                             build_filename("interactive", REQUEST_STOP_MODE, STOP_PROBABILITY, direction_tag, "html"))
    report_path = os.path.join(RESULTS_DIR,
                               build_filename("report", REQUEST_STOP_MODE, STOP_PROBABILITY, direction_tag, "txt"))

    print("\nGenerating outputs...")
    plot_speed(history, stops, track.stations, stats, title=f"Speed Profile | {base_title}", filepath=speed_path)
    plot_energy(history, stops, track.stations, stats, title=f"Energy Profile | {base_title}", filepath=energy_path)
    plot_interactive(history, stops, track.stations, stats, title=f"Simulation | {base_title}", filepath=html_path)

    generate_text_report(stats, stats_all, stats_none, stops, track.stations, report_path)
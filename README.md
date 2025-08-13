# Pebble-bed Seasonal Thermal Storage (example)

This folder contains:
- `pebble_optimize.py` — optimizer that **chooses nominal flow** by scanning charge ΔT targets and picks bed geometry (D, L) + pebbles (d_p, ε).
- `pebble_simulation.py` — **lumped** seasonal simulator that reads the optimized JSON and uses the chosen nominal flow.
- `pebble_1d_stratified.py` — **1D** packed-bed simulator with spatial stratification along the flow direction.
- Outputs:
  - `optimized_config_v2.json` — best design and the **chosen nominal flow** (mass & volumetric, velocities).
  - `pebble_bed_timeseries_v2.csv`, `fig_bed_temperature_v2.png`, `fig_power_flows_v2.png`
  - `pebble_1d_timeseries.csv`, `fig_1d_outlet_temp.png`

## How flow is chosen
The optimizer scans a small set of **charge ΔT targets** (default: 10–30 K). For each ΔT:
\dot m = Q_source / (c_p ΔT)
It evaluates UA, NTU, Ergun ΔP, fan power and seasonal energy, then **selects the ΔT (and thus flow)** that maximizes **net recovered energy** subject to a ΔP limit and simple shape constraints.

The chosen flow appears in the JSON as:
- `m_dot_nom_kg_s` (kg/s)
- `Vdot_m3_s` (m^3/s) and CFM (convert by ×2118.88)
- `superficial_velocity_m_s` and `interstitial_velocity_m_s`
- Also stored: `deltaT_air_charge_K` actually chosen by the optimizer.

## Edit points
- Open `pebble_optimize.py` and tweak `Scenario`:
  - **Loads & schedule**: `Q_source_kW`, `charge_days`, `charge_hours_per_day`, `storage_days`, `discharge_days`, `discharge_hours_per_day`
  - **Temperatures**: `T_ground_C`, `T_charge_in_C`, `T_discharge_in_C`
  - **ΔT candidates for flow choice**: `deltaT_air_charge_candidates_K`
  - **Losses**: `insulation_U_W_m2K`
  - **Design space**: `D_candidates_m`, `L_candidates_m`, `dp_candidates_m`, `eps_candidates`
  - **Limits**: `DELTA_P_LIMIT` (inside file), `FAN_EFF`
- Run optimizer → it writes `optimized_config_v2.json`.
- Run either simulator → it reads the JSON and reports the chosen flow and seasonal metrics.

## Physics
- Convective coefficient in a packed bed (Wakao–Kaguei): `Nu = 2 + 1.1 Re^0.6 Pr^(1/3)`, `h = Nu k / d_p`
- Specific area: `a = 6(1-ε) / d_p`; contact `UA = h a V_bed`
- Pressure drop: **Ergun**; fan: `P = ΔP·V̇ / η`
- Capacity: `Qcap = (1-ε) ρ_rock c_rock V_bed (T_hot - T_ground)`
- Losses: `Qloss = U_loss A_ext (T_bed - T_ground)`
- Lumped simulator: single bed temperature with per-step `min(UAΔT, ṁc_pΔT)`
- 1D simulator: 40 segments, upwind sweep each step; per-segment `min(UA_segΔT, ṁc_pΔT)`, no-flow dwell has only ground losses.

## Notes
- Results are for **scoping**. For design, use site-specific data, validate material properties, and consider a higher-fidelity model including moisture, by-pass, ducts, and true ground conduction.

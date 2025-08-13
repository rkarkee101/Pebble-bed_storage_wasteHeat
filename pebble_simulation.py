
import json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AIR_CP = 1005.0
AIR_RHO = 1.15

with open("/mnt/data/optimized_config_v2.json","r") as f:
    cfg = json.load(f)

scn = cfg["scenario"]
best = cfg["best_design"]

# Unpack
T_ground = scn["T_ground_C"]
T_charge_in = scn["T_charge_in_C"]
T_discharge_in = scn["T_discharge_in_C"]
Q_source_kW = scn["Q_source_kW"]
charge_days = scn["charge_days"]
charge_hours_per_day = scn["charge_hours_per_day"]
discharge_days = scn["discharge_days"]
discharge_hours_per_day = scn["discharge_hours_per_day"]
storage_days = scn["storage_days"]

# Geometry & UA
D = best["D_m"]; L = best["L_m"]; dp = best["dp_m"]; eps = best["eps"]
UA_contact = best["UA_contact_W_perK"]
UA_loss = best["UA_loss_W_perK"]

# Flow & fan
m_dot = best["m_dot_nom_kg_s"]
Vdot = best["Vdot_m3_s"]
v_s = best["superficial_velocity_m_s"]
v_i = best["interstitial_velocity_m_s"]
Pfan_W = best["Pfan_W"]

# Bed capacity
def bed_volume(D_m, L_m): return math.pi*(D_m*0.5)**2*L_m
V_bed = bed_volume(D, L)
m_solid = (1.0-eps)*2600.0*V_bed
C_bed = m_solid*800.0

# Time grid
dt = 900.0
def build_schedule():
    steps = []
    for day in range(charge_days):
        on = int((charge_hours_per_day*3600.0)/dt)
        off = int(((24-charge_hours_per_day)*3600.0)/dt)
        steps += [("charge", T_charge_in)]*on + [("idle", None)]*off
    for day in range(storage_days):
        steps += [("idle", None)]*int(24*3600.0/dt)
    for day in range(discharge_days):
        on = int((discharge_hours_per_day*3600.0)/dt)
        off = int(((24-discharge_hours_per_day)*3600.0)/dt)
        steps += [("discharge", T_discharge_in)]*on + [("idle", None)]*off
    return steps

schedule = build_schedule()

# Helpers
def charge_heat_rate_W(Tin, Tbed, Q_source_W):
    dT = max(0.0, Tin - Tbed)
    max_by_contact = UA_contact*dT
    max_by_air = (m_dot*AIR_CP)*dT
    return min(Q_source_W, max_by_contact, max_by_air)

def discharge_heat_rate_W(Tin, Tbed):
    dT = max(0.0, Tbed - Tin)
    max_by_contact = UA_contact*dT
    max_by_air = (m_dot*AIR_CP)*dT
    return min(max_by_contact, max_by_air)

# Simulation
T_bed = T_ground
energy_stored_J = 0.0
energy_lost_J = 0.0
energy_recovered_J = 0.0
fan_energy_J = 0.0

t_hours = []; T_bed_series = []; charge_kW=[]; discharge_kW=[]; loss_kW=[]; T_supply=[]
time_s=0.0
for mode, Tin in schedule:
    Q_loss_W = UA_loss*max(0.0, (T_bed - T_ground))

    if mode=="charge":
        Qin = charge_heat_rate_W(Tin, T_bed, Q_source_kW*1000.0)
        dE = (Qin - Q_loss_W)*dt
        T_bed += dE/max(C_bed,1e-12)
        energy_stored_J += max(0.0, Qin*dt)
        fan_energy_J += Pfan_W*dt
        charge_kW.append(Qin/1000.0); discharge_kW.append(0.0); T_supply.append(None)

    elif mode=="discharge":
        Qout = discharge_heat_rate_W(Tin, T_bed)
        dE = (-Qout - Q_loss_W)*dt
        T_bed += dE/max(C_bed,1e-12)
        Ts = Tin + Qout/(m_dot*AIR_CP + 1e-12)
        energy_recovered_J += max(0.0, Qout*dt)
        fan_energy_J += Pfan_W*dt
        charge_kW.append(0.0); discharge_kW.append(Qout/1000.0); T_supply.append(Ts)

    else:
        dE = (-Q_loss_W)*dt
        T_bed += dE/max(C_bed,1e-12)
        charge_kW.append(0.0); discharge_kW.append(0.0); T_supply.append(None)

    loss_kW.append(Q_loss_W/1000.0)
    T_bed_series.append(T_bed)
    t_hours.append(time_s/3600.0)
    time_s += dt

df = pd.DataFrame({"t_hours":t_hours,"T_bed_C":T_bed_series,"charge_kW":charge_kW,"discharge_kW":discharge_kW,"loss_kW":loss_kW,"T_supply_C":T_supply})
df.to_csv("/mnt/data/pebble_bed_timeseries_v2.csv", index=False)

# Summary
charge_hours = charge_days*charge_hours_per_day
discharge_hours = discharge_days*discharge_hours_per_day
Q_avail_MWh = Q_source_kW*charge_hours/1000.0
summary = {
  "Nominal mass flow (both phases)": f"{m_dot:.2f} kg/s",
  "Volumetric flow": f"{Vdot:.2f} m^3/s  (~{Vdot*2118.88:.0f} cfm)",
  "Superficial velocity": f"{v_s:.3f} m/s",
  "Interstitial velocity": f"{v_i:.3f} m/s",
  "Chosen ΔT at charge (optimizer)": f"{best['deltaT_air_charge_K']:.1f} K",
  "Pressure drop @ nominal": f"{best['deltaP_Pa']:.0f} Pa",
  "Fan power @ nominal": f"{Pfan_W/1000.0:.2f} kW",
  "Charge energy available (summer)": f"{Q_avail_MWh:.1f} MWhₜ",
  "Theoretical bed capacity": f"{best['Qcap_MWh']:.2f} MWhₜ",
  "Energy stored by end of charge": f"{(df.loc[df['t_hours']<=charge_days*24,'charge_kW'].sum()*(df['t_hours'].iloc[1]-df['t_hours'].iloc[0])/1000.0):.2f} MWhₜ",
  "Loss during storage dwell": "",
  "Total recovered during discharge": f"{(df.loc[df['t_hours']> (charge_days+storage_days)*24,'discharge_kW'].sum()*(df['t_hours'].iloc[1]-df['t_hours'].iloc[0])/1000.0):.2f} MWhₜ",
  "Net recovered minus fan": f"{((df['discharge_kW'].sum() - 0.0)* (df['t_hours'].iloc[1]-df['t_hours'].iloc[0]) /1000.0) - (Pfan_W*(charge_hours+discharge_hours)/3.6e6):.2f} MWhₜ"
}
# Loss during dwell
dt_h = df["t_hours"].iloc[1]-df["t_hours"].iloc[0]
mask_dwell = (df["t_hours"]>charge_days*24) & (df["t_hours"]<= (charge_days+storage_days)*24)
summary["Loss during storage dwell"] = f"{(df.loc[mask_dwell,'loss_kW'].sum()*dt_h/1000.0):.2f} MWhₜ"

# Plots
plt.figure()
plt.plot(df["t_hours"], df["T_bed_C"])
plt.xlabel("Time [h]"); plt.ylabel("Bed temperature [°C]"); plt.title("Bed temperature over the year (lumped)"); plt.tight_layout()
plt.savefig("/mnt/data/fig_bed_temperature_v2.png", dpi=160)

plt.figure()
plt.plot(df["t_hours"], df["charge_kW"], label="Charge")
plt.plot(df["t_hours"], df["discharge_kW"], label="Discharge")
plt.plot(df["t_hours"], df["loss_kW"], label="Loss")
plt.xlabel("Time [h]"); plt.ylabel("Power [kW]"); plt.title("Charge/Discharge/Loss power (lumped)"); plt.legend(); plt.tight_layout()
plt.savefig("/mnt/data/fig_power_flows_v2.png", dpi=160)

print("=== Lumped simulator (v2) summary ===")
for k,v in summary.items(): print(f"- {k}: {v}")
print("\nFiles: /mnt/data/pebble_bed_timeseries_v2.csv, fig_bed_temperature_v2.png, fig_power_flows_v2.png")

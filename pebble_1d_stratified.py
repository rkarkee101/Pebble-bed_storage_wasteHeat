
import json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AIR_CP = 1005.0
AIR_RHO = 1.15
ROCK_RHO = 2600.0
ROCK_CP  = 800.0

with open("/mnt/data/optimized_config_v2.json","r") as f:
    cfg = json.load(f)

scn = cfg["scenario"]
best = cfg["best_design"]

# Scenario
T_ground = scn["T_ground_C"]
T_charge_in = scn["T_charge_in_C"]
T_discharge_in = scn["T_discharge_in_C"]
Q_source_kW = scn["Q_source_kW"]
charge_days = scn["charge_days"]
charge_hours_per_day = scn["charge_hours_per_day"]
discharge_days = scn["discharge_days"]
discharge_hours_per_day = scn["discharge_hours_per_day"]
storage_days = scn["storage_days"]

# Geometry
D = best["D_m"]; L = best["L_m"]; dp = best["dp_m"]; eps = best["eps"]
A_cross = math.pi*(D*0.5)**2
V_bed = A_cross*L

# UA from optimizer -> distribute per segment by volume
UA_total = best["UA_contact_W_perK"]
UA_loss_total = best["UA_loss_W_perK"]

# Flow from optimizer
m_dot = best["m_dot_nom_kg_s"]
Vdot = best["Vdot_m3_s"]
v_s = best["superficial_velocity_m_s"]
v_i = best["interstitial_velocity_m_s"]
Pfan_W = best["Pfan_W"]

# Discretization
N_seg = 40
dz = L/N_seg
V_seg = V_bed/N_seg
UA_seg = UA_total/N_seg

# Loss distribution by surface area (approx uniform here)
UA_loss_seg = UA_loss_total/N_seg

# Rock capacity per segment
m_solid_seg = (1.0-eps)*ROCK_RHO*V_seg
C_seg = m_solid_seg*ROCK_CP

# Time grid
dt = 900.0

def build_schedule():
    steps=[]
    for day in range(charge_days):
        on=int((charge_hours_per_day*3600.0)/dt); off=int(((24-charge_hours_per_day)*3600.0)/dt)
        steps += [("charge", T_charge_in)]*on + [("idle", None)]*off
    for day in range(storage_days):
        steps += [("idle", None)]*int(24*3600.0/dt)
    for day in range(discharge_days):
        on=int((discharge_hours_per_day*3600.0)/dt); off=int(((24-discharge_hours_per_day)*3600.0)/dt)
        steps += [("discharge", T_discharge_in)]*on + [("idle", None)]*off
    return steps

schedule = build_schedule()

# State: rock temperature profile (z=0 is inlet)
Tz = np.full(N_seg, T_ground, dtype=float)

# Storage
t_hours=[]; Tout_list=[]; Tin_list=[]; Tmid_list=[]; E_store_J=0.0; E_rec_J=0.0; E_loss_J=0.0; fan_J=0.0

for step,(mode,Tin) in enumerate(schedule):
    # Per-step ground losses (no flow or with flow)
    Qloss = np.sum(UA_loss_seg*np.maximum(0.0, Tz - T_ground))
    E_loss_J += Qloss*dt

    if mode=="charge":
        # Upstream to downstream sweep
        Tair = Tin
        for i in range(N_seg):
            dT = Tair - Tz[i]
            if dT <= 0.0:
                # Air not hotter than rock at this segment: no heat in this segment
                pass
            else:
                Qmax_contact = UA_seg*dT
                Qmax_air = m_dot*AIR_CP*dT
                Q = min(Qmax_contact, Qmax_air)
                # Update segment rock temperature (can't exceed current air temp)
                Tz[i] += (Q*dt)/C_seg
                if Tz[i] > Tair: Tz[i] = Tair
                # Air cools by Q = m_dot*cp*(Tair_in - Tair_out)
                Tair -= Q/(m_dot*AIR_CP + 1e-12)
            # add ground loss effect locally for the segment (already counted globally via Qloss)
        Tout = Tair
        E_store_J += max(0.0, (Tin - Tout)*m_dot*AIR_CP*dt)
        fan_J += Pfan_W*dt

    elif mode=="discharge":
        # Upstream to downstream sweep (air is colder than rock)
        Tair = Tin
        for i in range(N_seg):
            dT = Tz[i] - Tair
            if dT <= 0.0:
                pass
            else:
                Qmax_contact = UA_seg*dT
                Qmax_air = m_dot*AIR_CP*dT
                Q = min(Qmax_contact, Qmax_air)
                Tz[i] -= (Q*dt)/C_seg
                if Tz[i] < Tair: Tz[i] = Tair
                Tair += Q/(m_dot*AIR_CP + 1e-12)
        Tout = Tair
        E_rec_J += max(0.0, (Tout - Tin)*m_dot*AIR_CP*dt)
        fan_J += Pfan_W*dt

    else:  # idle (no flow), only ground losses relax the profile
        # Relax rock toward ground: explicit Euler
        for i in range(N_seg):
            dE = -UA_loss_seg*max(0.0, Tz[i]-T_ground)*dt
            Tz[i] += dE/C_seg
        Tout = None

    t_hours.append(step*dt/3600.0); Tout_list.append(Tout); Tin_list.append(Tin if mode!="idle" else None); Tmid_list.append(float(np.mean(Tz)))

df = pd.DataFrame({"t_hours":t_hours,"T_out_C":Tout_list,"T_in_C":Tin_list,"T_bed_avg_C":Tmid_list})
df.to_csv("/mnt/data/pebble_1d_timeseries.csv", index=False)

# Plot outlet temperature
plt.figure()
plt.plot(df["t_hours"], df["T_out_C"])
plt.xlabel("Time [h]"); plt.ylabel("Outlet air temperature [°C]"); plt.title("1D packed-bed outlet temperature")
plt.tight_layout()
plt.savefig("/mnt/data/fig_1d_outlet_temp.png", dpi=160)

# Print summary
charge_hours = charge_days*charge_hours_per_day
discharge_hours = discharge_days*discharge_hours_per_day
Q_avail_MWh = Q_source_kW*charge_hours/1000.0
print("=== 1D stratified simulator summary ===")
print(f"- Nominal mass flow: {m_dot:.2f} kg/s  |  Vol flow: {Vdot:.2f} m^3/s (~{Vdot*2118.88:.0f} cfm)")
print(f"- Superficial velocity: {v_s:.3f} m/s  |  Interstitial: {v_i:.3f} m/s")
print(f"- ΔT at charge (chosen): {best['deltaT_air_charge_K']:.1f} K")
print(f"- Charge energy available (summer): {Q_avail_MWh:.1f} MWhₜ")
print(f"- Energy stored by end of charge (approx): {E_store_J/3.6e9:.2f} MWhₜ")
print(f"- Recovered during discharge: {E_rec_J/3.6e9:.2f} MWhₜ")
print(f"- Fan energy over season: {fan_J/3.6e9:.2f} MWhₜ")
print("Files: /mnt/data/pebble_1d_timeseries.csv, fig_1d_outlet_temp.png")

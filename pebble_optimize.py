
import json, math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple
import numpy as np

# --- Constants ---
AIR_CP = 1005.0
AIR_PR = 0.71
AIR_K  = 0.028
AIR_RHO = 1.15
AIR_MU  = 1.9e-5

ROCK_RHO = 2600.0
ROCK_CP  = 800.0

FAN_EFF = 0.6
DELTA_P_LIMIT = 1000.0  # Pa

@dataclass
class Scenario:
    # Temperatures (°C)
    T_ground_C: float = 18.0
    T_charge_in_C: float = 45.0
    T_discharge_in_C: float = 15.0
    T_supply_target_C: float = 35.0

    # Loads & schedules
    Q_source_kW: float = 150.0
    charge_days: int = 110
    charge_hours_per_day: int = 8
    discharge_days: int = 60
    discharge_hours_per_day: int = 8
    storage_days: int = 60

    # Loss / insulation
    insulation_U_W_m2K: float = 0.08

    # FLOW CHOICE: the optimizer will scan these ΔT targets to pick nominal flow
    deltaT_air_charge_candidates_K: Tuple[float, ...] = (10.0, 15.0, 20.0, 25.0, 30.0)

    # Search ranges
    D_candidates_m: Tuple[float, ...] = (4.0, 5.0, 6.0, 7.0, 8.0)
    L_candidates_m: Tuple[float, ...] = (4.0, 6.0, 8.0, 10.0)
    dp_candidates_m: Tuple[float, ...] = (0.03, 0.04, 0.05, 0.06)
    eps_candidates:   Tuple[float, ...] = (0.36, 0.40, 0.44)

def cross_area(D_m: float) -> float:
    return math.pi*(D_m*0.5)**2

def bed_volume(D_m: float, L_m: float) -> float:
    return cross_area(D_m)*L_m

def bed_surface_area(D_m: float, L_m: float) -> float:
    return math.pi*D_m*L_m + 2.0*cross_area(D_m)

def wakao_h(dp_m: float, eps: float, m_dot_kg_s: float, D_m: float) -> float:
    A = cross_area(D_m)
    v_s = m_dot_kg_s/(AIR_RHO*A)
    v_int = v_s/max(eps,1e-6)
    Re = AIR_RHO*v_int*dp_m/AIR_MU
    Nu = 2.0 + 1.1*(Re**0.6)*(AIR_PR**(1.0/3.0))
    h = Nu*AIR_K/max(dp_m,1e-9)
    return h

def specific_area_per_volume(dp_m: float, eps: float) -> float:
    return 6.0*(1.0-eps)/max(dp_m,1e-9)

def ergun_delta_p(dp_m: float, eps: float, m_dot_kg_s: float, D_m: float, L_m: float) -> float:
    A = cross_area(D_m)
    v_s = m_dot_kg_s/(AIR_RHO*A)
    term1 = 150.0*((1.0-eps)**2)*AIR_MU*v_s/( (eps**3)*max(dp_m,1e-9)**2 )
    term2 = 1.75*(1.0-eps)*AIR_RHO*(v_s**2)/( (eps**3)*max(dp_m,1e-9) )
    dP_per_L = term1 + term2
    return dP_per_L*L_m

def fan_power_W(deltaP_Pa: float, m_dot_kg_s: float) -> float:
    Vdot_m3_s = m_dot_kg_s/AIR_RHO
    return (deltaP_Pa*Vdot_m3_s)/FAN_EFF

def capacity_J(D_m: float, L_m: float, eps: float, T_hot_C: float, T_cold_C: float) -> float:
    V = bed_volume(D_m, L_m)
    m_solid = (1.0-eps)*ROCK_RHO*V
    return m_solid*ROCK_CP*max(0.0, (T_hot_C - T_cold_C))

def UA_contact_W_perK(dp_m: float, eps: float, m_dot_kg_s: float, D_m: float, L_m: float) -> float:
    a = specific_area_per_volume(dp_m, eps)
    h = wakao_h(dp_m, eps, m_dot_kg_s, D_m)
    V = bed_volume(D_m, L_m)
    return h*a*V

def optimize(scn: Scenario) -> Dict:
    charge_hours = scn.charge_days*scn.charge_hours_per_day
    discharge_hours = scn.discharge_days*scn.discharge_hours_per_day
    storage_hours = scn.storage_days*24
    Q_avail_J = scn.Q_source_kW*1000.0*charge_hours

    best = None
    best_score = -1e99

    for D in scn.D_candidates_m:
        for L in scn.L_candidates_m:
            A = cross_area(D)
            A_loss = bed_surface_area(D, L)

            for dp in scn.dp_candidates_m:
                for eps in scn.eps_candidates:
                    for dT in scn.deltaT_air_charge_candidates_K:
                        # Nominal flow from chosen ΔT
                        m_dot = (scn.Q_source_kW*1000.0)/(AIR_CP*max(dT,1e-6))
                        Vdot = m_dot/AIR_RHO
                        v_s = Vdot/max(A,1e-9)
                        v_i = v_s/max(eps,1e-6)

                        # UAs
                        UA = UA_contact_W_perK(dp, eps, m_dot, D, L)
                        UA_loss = scn.insulation_U_W_m2K*A_loss

                        NTU = UA/(AIR_CP*m_dot + 1e-9)

                        # ΔP & fan
                        dP = ergun_delta_p(dp, eps, m_dot, D, L)
                        Pfan_W = fan_power_W(dP, m_dot)

                        # Capacity & upper bounds
                        Qcap_J = capacity_J(D, L, eps, scn.T_charge_in_C, scn.T_ground_C)
                        eta_contact = NTU/(1.0 + NTU)
                        Qstore_possible_J = min(Q_avail_J*eta_contact, Qcap_J)

                        # Seasonal dwell loss estimate
                        frac = (Qstore_possible_J/(Qcap_J + 1e-12)) if Qcap_J > 0 else 0.0
                        T_avg_hot = scn.T_ground_C + frac*0.5*(scn.T_charge_in_C - scn.T_ground_C)
                        Qloss_J = UA_loss*(T_avg_hot - scn.T_ground_C)*storage_hours*3600.0
                        Q_after_storage_J = max(0.0, Qstore_possible_J - Qloss_J)

                        eta_discharge = eta_contact
                        Q_recovered_J = Q_after_storage_J*eta_discharge

                        Efan_J = Pfan_W*(charge_hours + discharge_hours)

                        net_J = Q_recovered_J - Efan_J

                        # Constraints and mild penalties
                        if dP > DELTA_P_LIMIT:
                            net_J -= 1e12
                        AR = L/max(D,1e-9)
                        if AR < 0.5 or AR > 3.0:
                            net_J -= 1e10

                        if net_J > best_score:
                            best_score = net_J
                            best = dict(
                                D_m=D, L_m=L, dp_m=dp, eps=eps,
                                deltaT_air_charge_K=dT,
                                m_dot_nom_kg_s=m_dot,
                                Vdot_m3_s=Vdot,
                                superficial_velocity_m_s=v_s,
                                interstitial_velocity_m_s=v_i,
                                UA_contact_W_perK=UA,
                                UA_loss_W_perK=UA_loss,
                                NTU=NTU,
                                deltaP_Pa=dP,
                                Pfan_W=Pfan_W,
                                Qcap_MWh=Qcap_J/3.6e9,
                                Q_avail_MWh=Q_avail_J/3.6e9,
                                Qstore_possible_MWh=Qstore_possible_J/3.6e9,
                                Qloss_MWh=Qloss_J/3.6e9,
                                Q_after_storage_MWh=Q_after_storage_J/3.6e9,
                                Q_recovered_MWh=Q_recovered_J/3.6e9,
                                Efan_MWh=Efan_J/3.6e9,
                                net_MWh=net_J/3.6e9
                            )

    result = {"scenario": asdict(scn), "best_design": best, "score_net_MWh": best_score/3.6e9}
    with open("/mnt/data/optimized_config_v2.json","w") as f:
        json.dump(result, f, indent=2)
    return result

if __name__ == "__main__":
    out = optimize(Scenario())
    print(json.dumps(out, indent=2))

"""Simulated complication trajectories after type 2 diabetes diagnosis (Markov process)."""
import numpy as np
import pandas as pd

COMPLICATIONS = ["Hypertension", "Dyslipidemia", "Neuropathy", "Retinopathy", "Nephropathy", "Coronary heart disease",
                 "Peripheral artery disease", "Stroke", "Foot ulcer", "Heart failure", "End-stage renal disease", "Amputation"]
# preferred successors (higher weight = more likely next complication)
NEXT = {"Start": {"Hypertension": 5, "Dyslipidemia": 4, "Neuropathy": 2, "Retinopathy": 1.5},
        "Hypertension": {"Dyslipidemia": 3, "Nephropathy": 3, "Coronary heart disease": 2.5, "Stroke": 1.2, "Retinopathy": 1.5},
        "Dyslipidemia": {"Hypertension": 3, "Coronary heart disease": 2.5, "Peripheral artery disease": 1.5, "Stroke": 1},
        "Neuropathy": {"Foot ulcer": 3, "Retinopathy": 2, "Nephropathy": 1.5, "Peripheral artery disease": 1},
        "Retinopathy": {"Nephropathy": 3, "Neuropathy": 2, "Hypertension": 1},
        "Nephropathy": {"End-stage renal disease": 2, "Heart failure": 1.5, "Coronary heart disease": 1.5, "Retinopathy": 1},
        "Coronary heart disease": {"Heart failure": 3, "Stroke": 1.5, "Peripheral artery disease": 1},
        "Peripheral artery disease": {"Foot ulcer": 2.5, "Amputation": 1, "Coronary heart disease": 1.5},
        "Foot ulcer": {"Amputation": 2.5, "Peripheral artery disease": 1},
        "Stroke": {"Heart failure": 1}, "Heart failure": {"End-stage renal disease": .5}, "End-stage renal disease": {}, "Amputation": {}}


def simulate(n=4000, seed=71):
    r = np.random.default_rng(seed)
    rows = []
    for pid in range(n):
        sex = r.choice(["Female", "Male"])
        age = int(np.clip(r.normal(56, 10), 30, 85))
        risk = r.lognormal(0, .35) * (1.15 if sex == "Male" else 1) * (1 + (age - 56) / 60)
        state, t, seen = "Start", 0.0, set()
        for _ in range(6):
            opts = {k: v for k, v in NEXT.get(state, {}).items() if k not in seen}
            if not opts or r.random() < .18 / risk:
                break
            keys = list(opts)
            p = np.array([opts[k] for k in keys], dtype=float); p /= p.sum()
            state = r.choice(keys, p=p)
            t += r.gamma(2, 1.6 / risk)
            if t > 15:
                break
            seen.add(state)
            rows.append((f"P{pid:05d}", sex, age, state, round(t, 2)))
    df = pd.DataFrame(rows, columns=["patient_id", "sex", "age_at_diagnosis", "complication", "years_since_diagnosis"])
    df["order"] = df.groupby("patient_id").cumcount() + 1
    return df

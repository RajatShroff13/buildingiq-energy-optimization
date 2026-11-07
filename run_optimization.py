"""
run_optimization.py
-------------------
Called by server.py with a path to new sensor CSV.
Runs XGBoost prediction + MOGA on that data.
Returns a results dict (JSON-serializable).

Usage from CLI:
    python3 run_optimization.py --input /path/to/new_sensors.csv

Usage from Python:
    from run_optimization import run
    results = run("/path/to/new_sensors.csv")
"""

import argparse, json, sys, os, warnings
from pathlib import Path
import joblib
import pandas as pd
import pandas as _pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

BASE        = str(Path(__file__).resolve().parent / "data" / "cleaned")
MODEL_PATH  = Path(__file__).resolve().parent / "models" / "model.pkl"
BATTERY_CAP = 20.0
BAT_MAX_CH  = 5.0
BAT_MAX_DI  = 5.0
BAT_EFF     = 0.92
TEMP_MIN    = 20.0
TEMP_MAX    = 26.0
N_HOURS     = 24
CHROM_LEN   = N_HOURS * 3
POP_SIZE    = 100
N_GEN       = 60

# ── helpers ──────────────────────────────────────────────────────────────────

def tou_price(hour, month, is_weekend):
    summer = month in [6, 7, 8, 9]
    if summer and not is_weekend and 14 <= hour <= 19:
        return 0.184
    elif summer:
        return 0.082
    elif not is_weekend and 6 <= hour <= 22:
        return 0.098
    return 0.062


def compute_pmv(ta, rh, va=0.1, met=1.2, clo=0.5):
    tr  = ta
    pa  = rh / 100 * np.exp(16.6536 - 4030.183 / (ta + 235))
    icl = 0.155 * clo
    m   = met * 58.15
    fcl = 1 + 1.29 * icl if icl <= 0.078 else 1.05 + 0.645 * icl
    hcf = 12.1 * np.sqrt(va)
    taa = ta + 273
    tra = tr + 273
    xn  = (ta + 273 + 35.5 - ta) / (3.5 * icl + 0.1) / 100
    xf  = xn
    p1  = icl * fcl; p2 = p1 * 3.96; p3 = p1 * 100; p4 = p1 * taa
    p5  = 308.7 - 0.028 * m + p2 * (tra / 100) ** 4
    for _ in range(150):
        xf  = (xn + xf) / 2
        hc  = max(hcf, 2.38 * abs(100 * xf - taa) ** 0.25)
        xn  = (p5 + p4 * hc - p2 * xf ** 4) / (100 + p3 * hc)
        if abs(xn - xf) < 0.00015:
            break
    tcl = 100 * xn - 273
    hl  = (3.05e-3 * (5733 - 6.99 * m - pa) +
           max(0, 0.42 * (m - 58.15)) +
           1.7e-5 * m * (5867 - pa) +
           0.0014 * m * (34 - ta) +
           3.96 * fcl * (xn ** 4 - (tra / 100) ** 4) +
           fcl * hc * (tcl - ta))
    pmv = (0.303 * np.exp(-0.036 * m) + 0.028) * (m - hl)
    return float(np.clip(pmv, -3, 3))


def add_features(df):
    """Add all features the model expects from raw sensor columns."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour"]        = pd.to_datetime(df["timestamp"]).dt.hour
    df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
    df["month"]       = pd.to_datetime(df["timestamp"]).dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # Occupancy proxy
    df["occupancy"] = (
        (df.get("avg_CO2_ppm", pd.Series(400, index=df.index)) > 650) |
        ((df["hour"].between(8, 18)) & (df["is_weekend"] == 0))
    ).astype(int)

    # TOU price
    df["tou_price_per_kwh"] = df.apply(
        lambda r: tou_price(r["hour"], r["month"], r["is_weekend"]), axis=1)

    # PMV
    def safe_pmv(r):
        try:
            return compute_pmv(r.get("avg_indoor_temp_C", 23),
                               r.get("avg_indoor_RH_pct", 50))
        except:
            return 0.0
    df["PMV"]        = df.apply(safe_pmv, axis=1)
    df["PPD"]        = (100 - 95 * np.exp(-0.03353 * df["PMV"]**4
                        - 0.2179 * df["PMV"]**2)).clip(5, 100)
    df["comfort_ok"] = df["PMV"].between(-0.5, 0.5).astype(int)

    # Cyclical
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Lags & rolling
    tgt = "total_load_kW"
    if tgt in df.columns:
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            df[f"load_lag_{lag}h"] = df[tgt].shift(lag)
        df["load_rolling_24h_mean"] = df[tgt].rolling(24, 1).mean()
        df["load_rolling_24h_std"]  = df[tgt].rolling(24, 1).std().fillna(0)
        df["load_rolling_7d_mean"]  = df[tgt].rolling(168, 1).mean()
    else:
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            df[f"load_lag_{lag}h"] = 0.5
        df["load_rolling_24h_mean"] = 0.5
        df["load_rolling_24h_std"]  = 0.1
        df["load_rolling_7d_mean"]  = 0.5

    sol = "solar_radiation_Wm2"
    if sol in df.columns:
        df["solar_lag_1h"]         = df[sol].shift(1).fillna(0)
        df["solar_lag_24h"]        = df[sol].shift(24).fillna(0)
        df["solar_rolling_3h_mean"]= df[sol].rolling(3, 1).mean()
    else:
        df["solar_lag_1h"] = df["solar_lag_24h"] = df["solar_rolling_3h_mean"] = 0

    df["temp_lag_1h"]  = df.get("outdoor_temp_C", pd.Series(15, index=df.index)).shift(1).fillna(15)
    df["temp_lag_24h"] = df.get("outdoor_temp_C", pd.Series(15, index=df.index)).shift(24).fillna(15)

    return df.ffill().fillna(0)


# ── main pipeline ─────────────────────────────────────────────────────────────

def run(input_path: str) -> dict:
    # 1. Load new data
    try:
        df = pd.read_csv(input_path, parse_dates=["timestamp"])
    except ValueError as exc:
        raise ValueError(
            "The CSV needs a 'timestamp' column that parses as a date "
            "(for example 2024-05-29 14:00:00)."
        ) from exc
    if df.empty:
        raise ValueError("The uploaded CSV has no rows.")
    df = add_features(df)

    # 2. Load scaler
    scaler = pd.read_csv(f"{BASE}/scaler_params.csv", index_col=0)

    # Feature order must match the trained model exactly, otherwise the model
    # quietly scores the wrong column in every slot.
    if MODEL_PATH.exists():
        payload = joblib.load(MODEL_PATH)
        model = payload["model"]
        FEATURES = list(payload["features"])
        tgt_min, tgt_rng = payload["tgt_min"], payload["tgt_rng"]
    else:
        model = None
        FEATURES = [
            "hour_sin","hour_cos","month_sin","month_cos","dow_sin","dow_cos",
            "is_weekend","occupancy","tou_price_per_kwh",
            "outdoor_temp_C","outdoor_RH_pct","solar_radiation_Wm2",
            "wind_speed_ms","pressure_mbar",
            "avg_indoor_temp_C","avg_indoor_RH_pct","avg_CO2_ppm",
            "pv_generation_kW","cooling_kW","heating_kW",
            "PMV","PPD","comfort_ok",
            "load_lag_1h","load_lag_2h","load_lag_3h","load_lag_6h",
            "load_lag_12h","load_lag_24h","load_lag_48h","load_lag_168h",
            "solar_lag_1h","solar_lag_24h","temp_lag_1h","temp_lag_24h",
            "load_rolling_24h_mean","load_rolling_24h_std",
            "load_rolling_7d_mean","solar_rolling_3h_mean",
        ]

    # Fill columns a raw sensor export does not carry. A neutral guess beats a
    # zero: zero solar or zero load is a real state, not "unknown".
    def _mean(col, fallback):
        return float(df[col].mean()) if col in df.columns else fallback

    defaults = {
        "lag": _mean("total_load_kW", 0.8),
        "solar": _mean("solar_radiation_Wm2", 0.0),
        "temp": _mean("outdoor_temp_C", 15.0),
        "rolling_mean": _mean("total_load_kW", 0.8),
        "rolling_std": float(df["total_load_kW"].std()) if "total_load_kW" in df.columns else 0.1,
    }
    for col in FEATURES:
        if col in df.columns:
            continue
        if col.startswith("load_lag"):
            df[col] = defaults["lag"]
        elif col.startswith("solar"):
            df[col] = defaults["solar"]
        elif col.startswith("temp_lag"):
            df[col] = defaults["temp"]
        elif col.endswith("_std"):
            df[col] = defaults["rolling_std"]
        elif col.startswith("load_rolling"):
            df[col] = defaults["rolling_mean"]
        else:
            df[col] = 0.0

    def scale_col(col_arr, col_name):
        if col_name in scaler.index:
            mn  = scaler.loc[col_name, "min"]
            rng = scaler.loc[col_name, "range"] or 1
            return (col_arr - mn) / rng
        return col_arr

    X = np.column_stack([scale_col(df[c].values, c) for c in FEATURES])

    # 3. Forecast. Fall back to a fresh fit only when no model file exists.
    if model is None:
        train_sc = pd.read_csv(f"{BASE}/train_scaled.csv")
        val_sc   = pd.read_csv(f"{BASE}/val_scaled.csv")
        y_train  = train_sc["total_load_kW"].values
        X_train  = train_sc[FEATURES].values
        y_val    = val_sc["total_load_kW"].values
        X_val    = val_sc[FEATURES].values

        model = xgb.XGBRegressor(
            n_estimators=800, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
            early_stopping_rounds=30, eval_metric="rmse"
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        tgt_min = scaler.loc["total_load_kW", "min"]
        tgt_rng = scaler.loc["total_load_kW", "range"]

    preds_scaled = model.predict(X)
    preds_kw     = preds_scaled * tgt_rng + tgt_min

    timestamps = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist()

    actual_kw = None
    metrics   = {}
    if "total_load_kW" in df.columns:
        actual_kw = df["total_load_kW"].tolist()
        rmse = float(np.sqrt(mean_squared_error(actual_kw, preds_kw)))
        mae  = float(mean_absolute_error(actual_kw, preds_kw))
        r2   = float(r2_score(actual_kw, preds_kw))
        metrics = {"rmse": round(rmse,4), "mae": round(mae,4), "r2": round(r2,4)}

    # 4. Run MOGA on LAST 24h
    last24      = df.tail(24)
    base_load   = preds_kw[-24:]
    solar_avail = last24.get("pv_generation_kW", pd.Series(0, index=last24.index)).values
    prices      = last24["tou_price_per_kwh"].values
    months      = last24["month"].values
    hours24     = last24["hour"].values

    def decode(chrom):
        sched = []
        for h in range(N_HOURS):
            sched.append({
                "hvac_sp":  TEMP_MIN + chrom[h*3+0] * (TEMP_MAX - TEMP_MIN),
                "lighting": float(np.clip(chrom[h*3+1], 0, 1)),
                "bat_act":  float(np.clip(chrom[h*3+2], -1, 1)),
            })
        return sched

    def simulate(chrom):
        sched = decode(chrom)
        soc = BATTERY_CAP * 0.5
        cost = energy = ppd_sum = 0.0
        for h in range(N_HOURS):
            s = sched[h]
            outdoor_t = 15 + 10 * np.sin(np.pi * h / 12)
            hvac_kw   = 0.15 * abs(s["hvac_sp"] - outdoor_t)
            light_kw  = 0.02 * s["lighting"]
            bat_kw    = 0.0
            if s["bat_act"] > 0:
                charge = min(s["bat_act"]*BAT_MAX_CH, BATTERY_CAP - soc)
                soc += max(charge,0) * BAT_EFF
                bat_kw = max(charge, 0)
            elif s["bat_act"] < 0:
                disch  = min(-s["bat_act"]*BAT_MAX_DI, soc)
                soc   -= max(disch, 0)
                bat_kw = -max(disch, 0)
            net = max(base_load[h] + hvac_kw + light_kw + bat_kw
                      - min(solar_avail[h], base_load[h]+hvac_kw+light_kw), 0)
            cost    += net * prices[h]
            energy  += net
            pmv_a    = (s["hvac_sp"] - 23.0) * 0.15
            ppd_a    = 100 - 95*np.exp(-0.03353*pmv_a**4 - 0.2179*pmv_a**2)
            ppd_sum += max(5, ppd_a)
        return [cost, energy, ppd_sum/N_HOURS]

    def dominates(a, b):
        return all(x<=y for x,y in zip(a,b)) and any(x<y for x,y in zip(a,b))

    def nds(objs):
        n = len(objs)
        dom = [[] for _ in range(n)]
        cnt = [0]*n
        fronts = [[]]
        for i in range(n):
            for j in range(n):
                if i==j: continue
                if dominates(objs[i], objs[j]): dom[i].append(j)
                elif dominates(objs[j], objs[i]): cnt[i]+=1
            if cnt[i]==0: fronts[0].append(i)
        k=0
        while fronts[k]:
            nxt=[]
            for i in fronts[k]:
                for j in dom[i]:
                    cnt[j]-=1
                    if cnt[j]==0: nxt.append(j)
            k+=1; fronts.append(nxt)
        ranks=[0]*n
        for fi,fr in enumerate(fronts):
            for idx in fr: ranks[idx]=fi
        return ranks, fronts

    pop = [np.clip(np.random.randn(CHROM_LEN)*0.3+0.5, 0, 1) for _ in range(POP_SIZE)]
    for gen in range(N_GEN):
        objs = [simulate(c) for c in pop]
        ranks, fronts = nds(objs)
        offspring = []
        while len(offspring) < POP_SIZE:
            i1,i2 = random.sample(range(POP_SIZE),2)
            i3,i4 = random.sample(range(POP_SIZE),2)
            p1 = pop[i1] if ranks[i1]<ranks[i2] else pop[i2]
            p2 = pop[i3] if ranks[i3]<ranks[i4] else pop[i4]
            pt = random.randint(1,CHROM_LEN-1)
            c1 = np.concatenate([p1[:pt],p2[pt:]])
            c2 = np.concatenate([p2[:pt],p1[pt:]])
            for c in [c1,c2]:
                mask = np.random.rand(CHROM_LEN) < 0.12
                c[mask] += np.random.normal(0,0.15,mask.sum())
                c[:] = np.clip(c,0,1)
            offspring.extend([c1,c2])
        combined = pop + offspring[:POP_SIZE]
        c_objs   = objs + [simulate(c) for c in offspring[:POP_SIZE]]
        r2_all, f2 = nds(c_objs)
        new_pop = []
        for fr in f2:
            if len(new_pop)+len(fr) <= POP_SIZE:
                new_pop.extend(fr)
            else:
                new_pop.extend(fr[:POP_SIZE-len(new_pop)])
                break
        pop = [combined[i] for i in new_pop]

    final_objs = [simulate(c) for c in pop]
    _, fronts_f = nds(final_objs)
    pareto_idx  = fronts_f[0]
    pareto      = [(pop[i], final_objs[i]) for i in pareto_idx]
    pareto.sort(key=lambda x: x[1][0])

    def pick(pref):
        if pref=="cheap":    return min(pareto, key=lambda x:x[1][0])
        if pref=="comfort":  return min(pareto, key=lambda x:x[1][2])
        cs=[s[1][0] for s in pareto]; es=[s[1][1] for s in pareto]; ps=[s[1][2] for s in pareto]
        mn_c,mx_c=min(cs),max(cs)+1e-9; mn_e,mx_e=min(es),max(es)+1e-9; mn_p,mx_p=min(ps),max(ps)+1e-9
        scores=[(((s[1][0]-mn_c)/(mx_c-mn_c)+(s[1][1]-mn_e)/(mx_e-mn_e)+(s[1][2]-mn_p)/(mx_p-mn_p)),s) for s in pareto]
        return min(scores,key=lambda x:x[0])[1]

    sol_A = pick("cheap")
    sol_B = pick("balanced")
    sol_C = pick("comfort")

    def sched_list(sol, label):
        rows=[]
        for h,s in enumerate(decode(sol[0])):
            rows.append({"hour":h,"hvac_setpoint_C":round(s["hvac_sp"],1),
                         "lighting":round(s["lighting"],2),
                         "battery_action":round(s["bat_act"],2),
                         "battery_mode":"charge" if s["bat_act"]>0.1 else ("discharge" if s["bat_act"]<-0.1 else "idle"),
                         "solution":label})
        return rows

    # build hourly arrays for charts
    def hourly_arrays(sol):
        sched = decode(sol[0])
        soc   = BATTERY_CAP*0.5
        net_loads,soc_arr,bat_arr=[],[],[]
        for h,s in enumerate(sched):
            outdoor_t = 15+10*np.sin(np.pi*h/12)
            hvac_kw   = 0.15*abs(s["hvac_sp"]-outdoor_t)
            light_kw  = 0.02*s["lighting"]
            bat_kw    = 0.0
            if s["bat_act"]>0:
                charge = min(s["bat_act"]*BAT_MAX_CH, BATTERY_CAP-soc)
                soc += max(charge,0)*BAT_EFF; bat_kw=max(charge,0)
            elif s["bat_act"]<0:
                disch = min(-s["bat_act"]*BAT_MAX_DI,soc)
                soc -= max(disch,0); bat_kw=-max(disch,0)
            net = max(base_load[h]+hvac_kw+light_kw+bat_kw-min(solar_avail[h],base_load[h]+hvac_kw+light_kw),0)
            net_loads.append(round(float(net),3))
            soc_arr.append(round(float(soc),2))
            bat_arr.append(round(float(s["bat_act"]),2))
        return net_loads,soc_arr,bat_arr

    nA,socA,batA = hourly_arrays(sol_A)
    nB,socB,batB = hourly_arrays(sol_B)
    nC,socC,batC = hourly_arrays(sol_C)


    # Deep-convert all numpy types for JSON serialization
    def _conv(obj):
        import numpy as np
        if isinstance(obj, dict):  return {k: _conv(v) for k,v in obj.items()}
        if isinstance(obj, list):  return [_conv(v) for v in obj]
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return obj


    # ── convert all numpy types for clean JSON ──
    import numpy as _np
    def _j(o):
        if isinstance(o, dict): return {k:_j(v) for k,v in o.items()}
        if isinstance(o, list): return [_j(v) for v in o]
        if isinstance(o, _np.integer): return int(o)
        if isinstance(o, _np.floating): return float(o)
        if isinstance(o, _np.ndarray): return o.tolist()
        return o

    return _j({
        "status":    "ok",
        "n_rows":    int(len(df)),
        "timestamps": timestamps,
        "predictions_kw": [round(float(v),4) for v in preds_kw],
        "actual_kw":  [round(float(v),4) for v in actual_kw] if actual_kw else None,
        "metrics":    metrics,
        "solar_kw":   [round(float(v),3) for v in solar_avail],
        "tou_prices": [round(float(v),3) for v in prices],
        "base_load":  [round(float(v),3) for v in base_load],
        "pareto": {
            "n_solutions": len(pareto),
            "costs":    [round(float(s[1][0]),3) for s in pareto],
            "energy":   [round(float(s[1][1]),3) for s in pareto],
            "discomfort":[round(float(s[1][2]),1) for s in pareto],
        },
        "schedules": {
            "A_cheapest":     {"objs": [round(float(v),3) for v in sol_A[1]], "hours": sched_list(sol_A,"A"), "net_load":nA,"soc":socA,"battery":batA},
            "B_balanced":     {"objs": [round(float(v),3) for v in sol_B[1]], "hours": sched_list(sol_B,"B"), "net_load":nB,"soc":socB,"battery":batB},
            "C_comfortable":  {"objs": [round(float(v),3) for v in sol_C[1]], "hours": sched_list(sol_C,"C"), "net_load":nC,"soc":socC,"battery":batC},
        },
        "pmv_series":    [round(float(v),3) for v in df["PMV"].tolist()],
        "co2_series":    [round(float(v),1) for v in df.get("avg_CO2_ppm",_pd.Series(400,index=df.index)).tolist()],
        "indoor_temp":   [round(float(v),2) for v in df.get("avg_indoor_temp_C",_pd.Series(23,index=df.index)).tolist()],
        "outdoor_temp":  [round(float(v),2) for v in df.get("outdoor_temp_C",_pd.Series(15,index=df.index)).tolist()],
    })


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast load and optimize a 24 h schedule.")
    parser.add_argument("--input", required=True, help="sensor CSV to analyse")
    parser.add_argument("--out", default=None, help="optional path for the JSON result")
    args = parser.parse_args()

    result = run(args.input)
    target = args.out or str(Path(__file__).resolve().parent / "outputs" / "last_run.json")
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"rows        : {result['n_rows']}")
    print(f"forecast    : {result['metrics'] or 'no measured column in input'}")
    print(f"pareto      : {result['pareto']['n_solutions']} solutions")
    for name, plan in result["schedules"].items():
        cost, energy, ppd = plan["objs"]
        print(f"  {name:<15} ${cost:.2f}  {energy:.1f} kWh  PPD {ppd:.1f}%")
    print(f"saved       : {target}")

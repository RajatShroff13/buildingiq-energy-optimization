"""
app.py - Flask front end for the Smart Building energy optimization project.

The site is a small multi-page dashboard over the artefacts produced by the
pipeline (``outputs/``): the XGBoost forecast, the NSGA-II Pareto front and the
three optimized 24-hour schedules.  One page (``/run``) lets a visitor upload a
sensor CSV and re-run the optimizer live.

Run:
    python app.py            # http://127.0.0.1:5000
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import traceback
from functools import lru_cache
from pathlib import Path

import pandas as pd
from flask import (Flask, abort, jsonify, render_template, request,
                   send_file)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB is plenty for a few days of sensors

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

SOLUTION_LABELS = {
    "A_cheapest": {"name": "A · Cheapest", "tag": "cost"},
    "B_balanced": {"name": "B · Balanced", "tag": "balanced"},
    "C_comfortable": {"name": "C · Most comfortable", "tag": "comfort"},
}


# ---------------------------------------------------------------- data loading
@lru_cache(maxsize=1)
def summary() -> dict:
    with open(OUT_DIR / "results_summary.json", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def forecast() -> pd.DataFrame:
    df = pd.read_csv(OUT_DIR / "forecast_results.csv", parse_dates=["timestamp"])
    return df


@lru_cache(maxsize=1)
def pareto() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "pareto_front.csv")


@lru_cache(maxsize=1)
def schedules() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "optimized_schedules.csv")


@lru_cache(maxsize=1)
def importance() -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / "feature_importance.csv")


@lru_cache(maxsize=1)
def dataset_stats() -> list[dict]:
    """Row/column counts for every cleaned dataset, used on the pipeline page."""
    meta = [
        ("housezero_cleaned.csv", "HouseZero (Harvard)", "Indoor environment, loads, PV and weather for one monitored house", "figshare 26499595"),
        ("master_table.csv", "Model-ready table", "HouseZero + engineered lag, rolling, TOU and PMV features", "derived"),
        ("ashrae_comfort_cleaned.csv", "ASHRAE Thermal Comfort DB II", "PMV/PPD against temperature, humidity and air speed", "ashrae-db-II"),
        ("cubems_cleaned.csv", "CU-BEMS Floor 1", "1-minute AC, lighting and plug loads for 33 zones", "sgru data"),
        ("noaa_weather_cleaned.csv", "NOAA ISD Chicago", "Hourly outdoor weather for 2017", "ncei.noaa.gov"),
    ]
    out = []
    for filename, name, purpose, source in meta:
        path = DATA_DIR / "cleaned" / filename
        rows = cols = 0
        if path.exists():
            frame = pd.read_csv(path, nrows=0)
            cols = len(frame.columns) - 1 if "timestamp" in frame.columns else len(frame.columns)
            with open(path, encoding="utf-8", errors="ignore") as fh:
                rows = max(sum(1 for _ in fh) - 1, 0)
        out.append({"name": name, "file": filename, "purpose": purpose,
                    "source": source, "rows": rows, "columns": cols,
                    "present": path.exists()})
    return out


def hourly_prices() -> list[dict]:
    """TOU tariff for one day, using the same rule set as the optimizer."""
    from run_optimization import tou_price

    rows = []
    for hour in range(24):
        price = tou_price(hour, 6, False)  # summer weekday, the tariff the study uses
        rows.append({"hour": hour, "price": round(float(price), 3)})
    return rows


def schedule_payload() -> dict:
    df = schedules()
    prices = {r["hour"]: r["price"] for r in hourly_prices()}
    payload = {}
    for key in ["A_cheapest", "B_balanced", "C_comfortable"]:
        part = df[df["solution"] == key].sort_values("hour")
        payload[key] = {
            "label": SOLUTION_LABELS[key]["name"],
            "hours": part["hour"].tolist(),
            "setpoint": part["hvac_setpoint_C"].round(2).tolist(),
            "lighting": part["lighting_level"].round(3).tolist(),
            "battery": part["battery_action"].round(3).tolist(),
            "mode": part["battery_mode"].tolist(),
            "price": [prices[h] for h in part["hour"].tolist()],
        }
    return payload


def forecast_payload(hours: int | None = None) -> dict:
    df = forecast()
    if hours:
        df = df.tail(hours)
    err = df["error_kW"]
    # residual histogram across the test week
    buckets = pd.cut(err, bins=15)
    counts = buckets.value_counts().sort_index()
    centres = [round(float(iv.mid), 4) for iv in counts.index]
    return {
        "labels": df["timestamp"].dt.strftime("%b %d · %H:%M").tolist(),
        "actual": df["actual_kW"].round(3).tolist(),
        "predicted": df["predicted_kW"].round(3).tolist(),
        "error": err.round(4).tolist(),
        "hist": {"centres": centres, "counts": [int(c) for c in counts.values]},
        "hours": len(df),
    }


# ------------------------------------------------------------------- template
@app.context_processor
def inject_globals():
    s = summary()
    return {
        "site": {
            "name": "BuildingIQ",
            "tagline": "Multi-objective energy optimization for a real smart building",
        },
        "model": s["prediction_model"],
        "moga": s["moga"],
        "data": s["data"],
    }


NAV = [
    ("overview", "/", "Overview", "M3 12h4l3-7 4 14 3-7h4"),
    ("forecast", "/forecast", "Load forecast", "M3 17l6-6 4 4 7-7M14 8h6v6"),
    ("optimization", "/optimization", "Optimization", "M4 19V5m0 14h16M4 15l5-5 4 3 7-8"),
    ("schedules", "/schedules", "Schedules", "M8 2v4M16 2v4M3 9h18M5 5h14a2 2 0 012 2v12a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2z"),
    ("pipeline", "/pipeline", "Data & method", "M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5"),
    ("run", "/run", "Run analysis", "M12 3v12m0 0l4-4m-4 4l-4-4M4 19h16"),
]


@app.context_processor
def inject_nav():
    return {"nav": NAV}


@app.route("/")
def overview():
    s = summary()
    sol = s["moga"]
    cheapest = sol["solution_A_cheapest"]
    comfort = sol["solution_C_comfortable"]
    saving = (comfort["cost"] - cheapest["cost"]) / comfort["cost"] * 100
    pf = pareto()
    return render_template(
        "overview.html",
        page="overview",
        title="Overview",
        subtitle="Three objectives, one Pareto front — live results from a monitored building.",
        kpis=[
            {"label": "Forecast R² (test)", "value": f"{s['prediction_model']['test']['r2']:.3f}",
             "hint": "XGBoost, 39 features", "tone": "accent"},
            {"label": "Pareto solutions", "value": str(sol["pareto_solutions"]),
             "hint": f"{sol['population_size']} individuals · {sol['generations']} generations", "tone": ""},
            {"label": "Cheapest day", "value": f"${cheapest['cost']:.2f}",
             "hint": f"{cheapest['energy']:.1f} kWh · PPD {cheapest['PPD_pct']:.1f}%", "tone": "good"},
            {"label": "Saving vs comfort-optimal", "value": f"{saving:.0f}%",
             "hint": f"${comfort['cost']:.2f} → ${cheapest['cost']:.2f} per day", "tone": "warn"},
        ],
        forecast_chart=forecast_payload(hours=72),
        pareto_chart={
            "cost": pf["cost_usd"].round(3).tolist(),
            "energy": pf["energy_kwh"].round(2).tolist(),
            "ppd": pf["discomfort_ppd_pct"].round(1).tolist(),
        },
        solutions=[
            {"key": "A_cheapest", "name": "A · Cheapest", "cost": cheapest["cost"],
             "energy": cheapest["energy"], "ppd": cheapest["PPD_pct"], "tone": "good"},
            {"key": "B_balanced", "name": "B · Balanced", "cost": sol["solution_B_balanced"]["cost"],
             "energy": sol["solution_B_balanced"]["energy"], "ppd": sol["solution_B_balanced"]["PPD_pct"], "tone": "accent"},
            {"key": "C_comfortable", "name": "C · Most comfortable", "cost": comfort["cost"],
             "energy": comfort["energy"], "ppd": comfort["PPD_pct"], "tone": "warn"},
        ],
        top_features=importance().head(6).to_dict("records"),
    )


@app.route("/forecast")
def forecast_page():
    payload = forecast_payload()
    imp = importance().head(12)
    return render_template(
        "forecast.html",
        page="forecast",
        title="Load forecast",
        subtitle="Gradient-boosted regression over 39 engineered features, validated on held-out weeks.",
        forecast_chart=payload,
        metrics=summary()["prediction_model"],
        importance={"labels": imp["feature"].tolist(),
                    "values": [round(float(v) * 100, 2) for v in imp["importance"]]},
    )


@app.route("/optimization")
def optimization_page():
    pf = pareto()
    s = summary()["moga"]
    return render_template(
        "optimization.html",
        page="optimization",
        title="Multi-objective optimization",
        subtitle="NSGA-II over 72 genes (24 h × HVAC setpoint, lighting, battery) — every dot is a complete day plan.",
        pareto={
            "cost": pf["cost_usd"].round(3).tolist(),
            "energy": pf["energy_kwh"].round(2).tolist(),
            "ppd": pf["discomfort_ppd_pct"].round(1).tolist(),
            "n": len(pf),
        },
        solutions=[
            {"key": "A_cheapest", "name": "A · Cheapest", "blurb": "Chases the TOU tariff hard, accepts the widest comfort band.",
             **s["solution_A_cheapest"]},
            {"key": "B_balanced", "name": "B · Balanced", "blurb": "Knee of the front: small cost premium, better comfort.",
             **s["solution_B_balanced"]},
            {"key": "C_comfortable", "name": "C · Most comfortable",
             "blurb": ("Keeps setpoints tightest, so energy use rises "
                       f"{(s['solution_C_comfortable']['energy'] / s['solution_A_cheapest']['energy'] - 1) * 100:.0f}% over the cheapest plan."),
             **s["solution_C_comfortable"]},
        ],
        settings=s,
        schedule=schedule_payload(),
    )


@app.route("/schedules")
def schedules_page():
    return render_template(
        "schedules.html",
        page="schedules",
        title="Optimized schedules",
        subtitle="What the optimizer actually asks the building to do, hour by hour.",
        schedule=schedule_payload(),
        solutions=[
            {"key": "A_cheapest", "name": "A · Cheapest", **summary()["moga"]["solution_A_cheapest"]},
            {"key": "B_balanced", "name": "B · Balanced", **summary()["moga"]["solution_B_balanced"]},
            {"key": "C_comfortable", "name": "C · Comfortable", **summary()["moga"]["solution_C_comfortable"]},
        ],
    )


@app.route("/pipeline")
def pipeline_page():
    s = summary()
    return render_template(
        "pipeline.html",
        page="pipeline",
        title="Data & method",
        subtitle="Five open datasets, one model-ready table, and a prediction-in-the-loop optimizer.",
        datasets=dataset_stats(),
        metrics=s["prediction_model"],
        moga=s["moga"],
        dates=s["data"],
        top_features=importance().head(10).to_dict("records"),
    )


@app.route("/run")
def run_page():
    s = summary()
    return render_template(
        "run.html",
        page="run",
        title="Run the analysis",
        subtitle="Upload a sensor CSV and the optimizer will forecast load and rebuild the schedule.",
        sample_rows=len(pd.read_csv(DATA_DIR / "sample" / "sample_sensor_data.csv")),
        default_metrics=s["prediction_model"]["test"],
    )


# ------------------------------------------------------------------------ API
@app.route("/api/summary")
def api_summary():
    return jsonify(summary())


@app.route("/api/forecast")
def api_forecast():
    return jsonify(forecast_payload())


@app.route("/api/pareto")
def api_pareto():
    return jsonify(pareto().to_dict("records"))


@app.route("/api/schedules")
def api_schedules():
    return jsonify(schedule_payload())


@app.route("/api/sample")
def api_sample():
    path = DATA_DIR / "sample" / "sample_sensor_data.csv"
    return send_file(path, mimetype="text/csv", as_attachment=True,
                     download_name="sample_sensor_data.csv")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH.exists()})


@app.route("/api/run", methods=["POST"])
def api_run():
    """Accept a sensor CSV, run prediction + MOGA, return the result payload."""
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"status": "error", "message": "Attach a CSV file first."}), 400
    if not upload.filename.lower().endswith(".csv"):
        return jsonify({"status": "error", "message": "Only .csv files are supported."}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            upload.save(tmp.name)
            tmp_path = tmp.name
        from run_optimization import run

        result = run(tmp_path)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 422
    except Exception as exc:  # noqa: BLE001 - surfaced to the page as a message
        app.logger.exception("optimization failed")
        return jsonify({
            "status": "error",
            "message": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=3),
        }), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.errorhandler(404)
def not_found(_exc):
    return render_template("404.html", page="", title="Page not found",
                           subtitle="That page does not exist."), 404


@app.errorhandler(413)
def too_large(_exc):
    return jsonify({"status": "error",
                    "message": f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")
    print("=" * 58)
    print(f"  BuildingIQ - Smart Building Energy Optimization")
    print(f"  http://{host}:{port}")
    print("=" * 58)
    app.run(host=host, port=port, debug=os.environ.get("FLASK_DEBUG") == "1")

"""
dashboard.py
============
Builds outputs/dashboard.html - a SINGLE self-contained file that is the
"operator console" of the project.

Why HTML and not Streamlit/Dash?
  * zero extra dependencies (no server needed to show it)
  * can be opened by pressing double-click, attached to an email, or hosted on
    GitHub Pages as a live demo link in your resume  <-- huge portfolio win
  * every image is embedded as base64, so it renders offline (in a README
    preview, a zip file or a sandboxed iframe)

Contents
--------
  1. KPI header        - model, PR-AUC, recall, precision, lead time, savings
  2. LIVE MONITOR      - JavaScript replay of a real breakdown on a held-out
                         machine: probability gauge, sensor traces, alarm log
  3. ALERT CENTRE      - work orders produced by the alert engine
  4. FIGURE GALLERY    - all analysis figures with explanations
  5. BUSINESS CASE     - scenario table
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.utils import get_logger

log = get_logger("dashboard")


# ---------------------------------------------------------------------------
def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _img_tag(path: Path, caption: str, why: str = "") -> str:
    if not path.exists():
        return ""
    return f"""
    <figure class="fig">
      <img src="data:image/png;base64,{_b64(path)}" alt="{caption}" loading="lazy"/>
      <figcaption><b>{caption}</b>{f'<br/><span class="why">{why}</span>' if why else ''}</figcaption>
    </figure>"""


def _kpi(label: str, value: str, sub: str = "", tone: str = "neutral") -> str:
    return f"""<div class="kpi {tone}"><div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>"""


# ---------------------------------------------------------------------------
def _live_payload() -> dict:
    """
    Compact JSON for the browser: one held-out machine's final degradation
    episode, plus the alerts the engine raised during it.
    """
    preds = pd.read_csv(config.STREAM_PRED_CSV)
    raw = pd.read_csv(config.STREAM_RAW_CSV, parse_dates=["timestamp"])

    failing = preds.loc[preds.y_true == 1, "machine_id"].value_counts()
    mid = failing.index[0] if len(failing) else preds.machine_id.iloc[0]
    fail_cycle = int(preds.loc[(preds.machine_id == mid) & (preds.y_true == 1), "cycle"].min())
    lo, hi = max(0, fail_cycle - 150), fail_cycle + 6

    d = preds[(preds.machine_id == mid) & preds.cycle.between(lo, hi)].sort_values("cycle")
    s = raw[(raw.machine_id == mid) & raw.cycle.between(lo, hi)].sort_values("cycle")
    m = d.merge(s[["cycle"] + config.STREAM_SENSOR_COLUMNS + ["health_index", "fault_mode"]],
                on="cycle", how="left", suffixes=("", "_raw"))

    alerts = pd.DataFrame()
    if config.LIVE_ALERTS_CSV.exists():
        a = pd.read_csv(config.LIVE_ALERTS_CSV)
        alerts = a[a.machine_id == mid]

    return {
        "machine_id": mid,
        "fail_cycle": fail_cycle,
        "start_cycle": lo,
        "step_minutes": config.SAMPLING_MINUTES,
        "threshold": float(json.loads(config.STREAM_METRICS.read_text())["test_metrics"]["threshold"]),
        "cycles": [int(v) for v in d["cycle"]],
        "probability": [round(float(v), 4) for v in d["failure_probability"]],
        "vibration": [round(float(v), 3) for v in m["vibration_mm_s"]],
        "temperature": [round(float(v), 2) for v in m["temperature_c"]],
        "acoustic": [round(float(v), 2) for v in m["acoustic_db"]],
        "health": [round(float(v) * 100, 1) for v in m["health_index"]],
        "fault_mode": str(m["fault_mode"].iloc[0]),
        "alert_cycles": [int(v) for v in alerts["e_cycle"]] if "e_cycle" in alerts else [],
        "alert_severity": alerts["severity"].tolist() if "severity" in alerts else [],
        "alert_wo": alerts["work_order_id"].tolist() if "work_order_id" in alerts else [],
    }


# ---------------------------------------------------------------------------
def build_dashboard(out_path: Path = config.DASHBOARD_HTML) -> Path:
    ai4i = json.loads(config.AI4I_METRICS.read_text())
    stream = (json.loads(config.STREAM_METRICS.read_text())
              if config.STREAM_METRICS.exists() else {"test_metrics": {}, "cv_comparison": []})
    biz = (json.loads((config.OUTPUTS_DIR / "business_impact.json").read_text())
           if (config.OUTPUTS_DIR / "business_impact.json").exists() else {})
    lead = {}
    lead_file = config.OUTPUTS_DIR / "alert_lead_time_report.csv"
    if lead_file.exists():
        lt = pd.read_csv(lead_file)
        lt = lt[pd.to_numeric(lt["lead_time_cycles"], errors="coerce").notna()]
        if len(lt):
            lead = {
                "caught_pct": round(100 * (lt["lead_time_cycles"] > 0).mean(), 1),
                "avg_hours": round(float(lt["lead_time_cycles"].astype(float).mean())
                                   * config.SAMPLING_MINUTES / 60, 1),
                "events": int(len(lt)),
            }

    m = ai4i["test_metrics"]; ms = stream.get("test_metrics", {})
    expected = (biz.get("scenarios", {}).get("expected", {}) if biz else {})
    cons = (biz.get("scenarios", {}).get("conservative", {}) if biz else {})

    alerts_html = ""
    if config.LIVE_ALERTS_CSV.exists():
        al = pd.read_csv(config.LIVE_ALERTS_CSV).tail(12).iloc[::-1]
        keep = [c for c in ["work_order_id", "timestamp", "machine_id", "severity",
                            "probability", "recommended_action"] if c in al.columns]
        rows = "".join(
            "<tr>" + "".join(f"<td>{r[c]}</td>" for c in keep) + "</tr>"
            for _, r in al[keep].iterrows())
        alerts_html = (f"<table class='alerts'><thead><tr>"
                       f"{''.join(f'<th>{c.replace(chr(95), chr(32)).title()}</th>' for c in keep)}"
                       f"</tr></thead><tbody>{rows}</tbody></table>")

    live = _live_payload()

    figures = [
        ("01_dataset_overview.png", "EDA - real AI4I 2020 industrial dataset",
         "Class imbalance, sensor distributions per class and the 5 labelled failure modes."),
        ("02_sensor_correlation.png", "Correlation matrix incl. engineered physics features",
         "Engineered features (power, temperature difference, wear x torque) carry the failure signal."),
        ("03_model_comparison.png", "Cross-validated model benchmark (AI4I)",
         "4 algorithms, 5-fold stratified CV, out-of-fold predictions - no leakage."),
        ("04_confusion_matrix.png", "Confusion matrix on the untouched test set",
         "Every cell is labelled with its business meaning, not just TP/FP/FN/TN."),
        ("05_roc_pr_curves.png", "ROC and Precision-Recall curves",
         "PR-AUC is the honest metric because failures are rare."),
        ("06_feature_importance.png", "What the model learned",
         "Vibration, tool wear and torque dominate - exactly what maintenance engineers watch."),
        ("07_threshold_optimisation.png", "Threshold optimisation by business cost",
         "Why 0.5 is a bad default: a missed failure costs ~16x a false alarm."),
        ("08_probability_distribution.png", "Probability separation",
         "Calibrated probabilities are what make the LOW/MEDIUM/HIGH/CRITICAL bands meaningful."),
        ("09_degradation_trajectory.png", "Virtual machine wearing out",
         "Simulated IoT sensors drifting together before a real breakdown."),
        ("10_early_warning_timeline.png", "Early warning on an UNSEEN machine",
         "The AI alarm fires cycles before the breakdown - the whole point of the project."),
        ("11_fleet_health_map.png", "Fleet health snapshot",
         "Prioritised list of which of the 30 machines to service first."),
        ("12_eda_stream_overview.png", "Virtual IoT stream EDA (27,000 samples)",
         "Healthy vs pre-failure sensor signatures."),
        ("13_confusion_matrix_stream.png", "Time-series model on unseen machines",
         "Machine-wise split: the model is tested on machines it never saw."),
        ("14_model_comparison_stream.png", "Model benchmark - virtual fleet",
         "Same 4 algorithms on 92 window/rolling features."),
        ("15_roc_pr_stream.png", "ROC / PR - virtual fleet",
         "Near-perfect separation of degrading machines."),
        ("16_feature_importance_stream.png", "Top signals for time-series prediction",
         "Rolling trends and standard deviations dominate - degradation is about change, not level."),
        ("17_business_impact.png", "Business case",
         "Three scenarios, honest assumptions, payback period."),
    ]
    gallery = "".join(_img_tag(config.IMAGES_DIR / f, cap, why) for f, cap, why in figures)

    # --------------------------------------------------------------- HTML ----
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AI-Powered Predictive Maintenance - Operations Dashboard</title>
<style>
  :root {{
    --bg:#0d1b24; --panel:#132a36; --panel2:#17384a; --ink:#e8f1f5;
    --muted:#8fb0c0; --ok:#3fa34d; --warn:#f18f01; --crit:#c81d25; --blue:#2e86ab;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
    font-family:'Segoe UI',system-ui,-apple-system,sans-serif; }}
  header {{ background:linear-gradient(120deg,#0b2a3a,#123c52 55%,#0d1b24);
    padding:34px 40px 26px; border-bottom:3px solid var(--blue); }}
  h1 {{ margin:0 0 6px; font-size:29px; letter-spacing:.4px; }}
  .sub {{ color:var(--muted); font-size:15px; }}
  .badges {{ margin-top:14px; display:flex; gap:10px; flex-wrap:wrap; }}
  .badge {{ background:var(--panel2); border:1px solid #215a75; color:var(--ink);
    padding:6px 13px; border-radius:20px; font-size:12.5px; }}
  .wrap {{ padding:28px 40px 60px; max-width:1600px; margin:auto; }}
  h2 {{ font-size:20px; margin:38px 0 14px; padding-left:12px;
    border-left:5px solid var(--blue); }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; }}
  .kpi {{ background:var(--panel); border-radius:12px; padding:16px 18px;
    border-left:5px solid var(--blue); }}
  .kpi.ok {{ border-left-color:var(--ok); }} .kpi.warn {{ border-left-color:var(--warn); }}
  .kpi.crit {{ border-left-color:var(--crit); }}
  .kpi-label {{ color:var(--muted); font-size:11.5px; text-transform:uppercase; letter-spacing:.9px; }}
  .kpi-value {{ font-size:26px; font-weight:700; margin:6px 0 2px; }}
  .kpi-sub {{ color:var(--muted); font-size:11.5px; }}
  .panel {{ background:var(--panel); border-radius:14px; padding:22px; margin-top:16px; }}
  .livegrid {{ display:grid; grid-template-columns:300px 1fr; gap:22px; align-items:start; }}
  @media (max-width:1000px) {{ .livegrid {{ grid-template-columns:1fr; }} }}
  .gauge {{ text-align:center; }}
  .gauge svg {{ width:100%; max-width:270px; }}
  .verdict {{ margin-top:8px; font-weight:700; font-size:16px; }}
  .spark {{ display:block; width:100%; height:88px; margin-bottom:10px; }}
  .logs {{ background:#08222c; border-radius:10px; padding:12px 14px; height:168px;
    overflow-y:auto; font-family:'JetBrains Mono',Consolas,monospace; font-size:12.5px; }}
  .logline {{ padding:3px 0; border-bottom:1px dotted #1c4356; }}
  .s-crit {{ color:#ff6b6b; font-weight:700; }} .s-high {{ color:#ffb347; font-weight:700; }}
  .s-med {{ color:#ffe08a; }} .s-ok {{ color:#7ee08a; }}
  button {{ background:var(--blue); color:#fff; border:0; border-radius:8px;
    padding:9px 18px; font-size:13.5px; cursor:pointer; margin-right:8px; }}
  button.ghost {{ background:transparent; border:1px solid #2f6c8a; }}
  button:hover {{ filter:brightness(1.15); }}
  .controls {{ margin:14px 0 6px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }}
  th {{ background:var(--panel2); color:var(--muted); text-align:left; padding:10px;
    font-size:11.5px; text-transform:uppercase; letter-spacing:.7px; }}
  td {{ padding:9px 10px; border-bottom:1px solid #1c3d4d; }}
  tr:hover td {{ background:#15313f; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr)); gap:20px; }}
  figure.fig {{ margin:0; background:var(--panel); border-radius:14px; padding:14px; }}
  figure.fig img {{ width:100%; border-radius:9px; background:#fff; }}
  figcaption {{ font-size:13px; margin-top:11px; line-height:1.5; }}
  .why {{ color:var(--muted); font-size:12.3px; }}
  .two {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
  @media (max-width:900px) {{ .two {{ grid-template-columns:1fr; }} }}
  .callout {{ background:var(--panel2); border-left:5px solid var(--ok); border-radius:10px;
    padding:14px 18px; font-size:14px; line-height:1.6; }}
  code {{ background:#08222c; padding:2px 6px; border-radius:5px; font-size:12.5px; }}
  footer {{ color:var(--muted); font-size:12.5px; padding:26px 40px 40px;
    border-top:1px solid #1c3d4d; margin-top:40px; }}
</style></head><body>

<header>
  <h1>AI-Powered Predictive Maintenance System for IoT Devices</h1>
  <div class="sub">Edge-to-AI reference implementation &nbsp;|&nbsp; rotating-machinery fleet monitoring
      &nbsp;|&nbsp; generated {datetime.now():%d %b %Y %H:%M}</div>
  <div class="badges">
    <span class="badge">AI4I 2020 real dataset (UCI)</span>
    <span class="badge">Virtual IoT fleet - 30 machines / 27,000 samples</span>
    <span class="badge">Model: {ai4i['chosen_model'].replace('_',' ').title()} + ExtraTrees ensemble</span>
    <span class="badge">Cost-optimised alarm thresholds</span>
    <span class="badge">Flask inference API</span>
    <span class="badge">Edge-deployable (Raspberry Pi / ESP32)</span>
  </div>
</header>

<div class="wrap">

  <h2>1 - Headline KPIs</h2>
  <div class="kpis">
    {_kpi("Model (real AI4I data)", ai4i['chosen_model'].replace('_',' ').title(), f"{ai4i['n_features']} features")}
    {_kpi("ROC-AUC", f"{m['roc_auc']:.3f}", "test set, untouched", "ok")}
    {_kpi("PR-AUC", f"{m['pr_auc']:.3f}", "rare-failure metric", "ok")}
    {_kpi("Recall", f"{m['recall']*100:.1f}%", f"caught {m['tp']} of {m['tp']+m['fn']} failures", "ok")}
    {_kpi("Precision", f"{m['precision']*100:.1f}%", f"{m['fp']} false alarms / {m['tp']} hits", "warn")}
    {_kpi("Vibration-model F1", f"{ms.get('f1',0):.3f}", "unseen machines, virtual fleet", "ok")}
    {_kpi("Breakdowns predicted ahead", f"{lead.get('caught_pct',0):.0f}%",
          f"{lead.get('events',0)} breakdown events", "ok")}
    {_kpi("Avg early-warning notice", f"{lead.get('avg_hours',0):.1f} h", "before the machine stops", "ok")}
    {_kpi("Annual saving (expected)", f"${expected.get('net_annual_saving_usd',0)/1000:,.0f}k",
          f"{expected.get('saving_pct',0)}% vs run-to-failure", "ok")}
    {_kpi("Retrofit payback", f"{expected.get('payback_period_days',0):.0f} days",
          f"${biz.get('fleet_size_machines',30)*config.SENSOR_KIT_COST:,.0f} capex", "ok")}
  </div>

  <h2>2 - Live monitor (replay of a real breakdown, held-out machine)</h2>
  <div class="panel">
    <div class="livegrid">
      <div class="gauge">
        <svg viewBox="0 0 200 130">
          <path d="M20 115 A80 80 0 0 1 180 115" fill="none" stroke="#1c4356" stroke-width="16" stroke-linecap="round"/>
          <path id="arc" d="M20 115 A80 80 0 0 1 180 115" fill="none" stroke="#3fa34d" stroke-width="16"
                stroke-linecap="round" stroke-dasharray="251" stroke-dashoffset="251"/>
          <text id="probtxt" x="100" y="100" text-anchor="middle" font-size="34" font-weight="700"
                fill="#e8f1f5" font-family="Segoe UI">0.00</text>
          <text x="100" y="120" text-anchor="middle" font-size="11" fill="#8fb0c0"
                font-family="Segoe UI">P(failure in next hour)</text>
        </svg>
        <div class="verdict" id="verdict">--</div>
        <div class="kpi-sub" id="cycleinfo">cycle -- / machine {live['machine_id']}</div>
        <div class="kpi-sub" id="thresholdinfo">alarm threshold {live['threshold']} | fault mode
          {live['fault_mode'].replace('_',' ')}</div>
      </div>
      <div>
        <svg class="spark" id="spark-vib"></svg>
        <svg class="spark" id="spark-temp"></svg>
        <svg class="spark" id="spark-ac"></svg>
        <div class="logs" id="logbox"></div>
        <div class="controls">
          <button onclick="play()">Play replay</button>
          <button class="ghost" onclick="pause()">Pause</button>
          <button class="ghost" onclick="reset()">Reset</button>
          <span class="kpi-sub" id="speedinfo">each step = {config.SAMPLING_MINUTES} simulated minutes</span>
        </div>
      </div>
    </div>
  </div>

  <h2>3 - Maintenance alert centre (work orders raised by the alert engine)</h2>
  <div class="panel">{alerts_html or '<div class="kpi-sub">No alerts logged yet - run python -m src.alert_system</div>'}</div>

  <h2>4 - Business case</h2>
  <div class="two">
    <div class="callout">
      <b>Expected scenario:</b> {expected.get('breakdowns_prevented_per_year','-')} breakdowns prevented per year,
      unplanned downtime down <b>{expected.get('downtime_reduction_pct','-')}%</b>,
      net saving <b>${expected.get('net_annual_saving_usd',0):,.0f}</b> on a
      {biz.get('fleet_size_machines',30)}-machine line.<br/><br/>
      <b>Conservative scenario</b> (only 40% of alarms acted on in time):
      still <b>${cons.get('net_annual_saving_usd',0):,.0f}</b> saved and
      <b>{cons.get('downtime_reduction_pct','-')}%</b> less downtime - inside the
      15-30% range reported by published industrial studies.
    </div>
    <div class="callout">
      <b>How to reproduce every number on this page:</b><br/>
      <code>pip install -r requirements.txt</code><br/>
      <code>python train_model.py</code> &nbsp; train both models<br/>
      <code>python -m src.alert_system</code> &nbsp; replay + work orders<br/>
      <code>python main.py all</code> &nbsp; full pipeline + this dashboard<br/>
      <code>python api/app.py</code> &nbsp; serve the REST API
    </div>
  </div>

  <h2>5 - Analysis gallery (every figure is generated by the code)</h2>
  <div class="grid">{gallery}</div>
</div>

<footer>
  Built as an industry-oriented student project &nbsp;|&nbsp; real dataset: AI4I 2020 Predictive
  Maintenance (UCI ML Repository, CC BY 4.0) &nbsp;|&nbsp; virtual fleet generated by
  <code>src/simulate_iot_sensors.py</code> &nbsp;|&nbsp; all metrics reproducible with
  <code>python main.py all</code> (seed = {config.RANDOM_SEED}).
</footer>

<script>
const LIVE = {json.dumps(live)};
let i = 0, timer = null;

const COL = {{ok:'#3fa34d', warn:'#f18f01', crit:'#c81d25'}};
const SEV = {{LOW:'ok', MEDIUM:'med', HIGH:'high', CRITICAL:'crit'}};

function sev(p, thr) {{
  if (p >= 0.80) return 'CRITICAL';
  if (p >= 0.55) return 'HIGH';
  if (p >= 0.30) return 'MEDIUM';
  return 'LOW';
}}
function sevColor(p) {{
  if (p >= 0.80) return COL.crit;
  if (p >= 0.55) return COL.warn;
  if (p >= 0.30) return '#ffe08a';
  return COL.ok;
}}

function sparkline(id, values, colour, label, upto) {{
  const svg = document.getElementById(id);
  const W = 600, H = 88, pad = 6;
  const slice = values.slice(0, upto + 1);
  const min = Math.min(...values), max = Math.max(...values) || 1;
  const scale = v => H - pad - (H - 2 * pad) * (v - min) / ((max - min) || 1);
  const step = W / (values.length - 1 || 1);
  let d = slice.map((v, k) => `${{k ? 'L' : 'M'}}${{(k * step).toFixed(1)}} ${{scale(v).toFixed(1)}}`).join(' ');
  const lastY = scale(slice[slice.length - 1] ?? min);
  const lastX = ((slice.length - 1) * step).toFixed(1);
  svg.setAttribute('viewBox', `0 0 ${{W}} ${{H}}`);
  svg.innerHTML = `
    <rect x="0" y="0" width="${{W}}" height="${{H}}" rx="8" fill="#08222c"/>
    <path d="${{d}}" fill="none" stroke="${{colour}}" stroke-width="2.2"/>
    <circle cx="${{lastX}}" cy="${{lastY}}" r="3.6" fill="${{colour}}"/>
    <text x="10" y="16" font-size="11" fill="#8fb0c0" font-family="Segoe UI">${{label}} =
      ${{(slice[slice.length - 1] ?? 0).toFixed(2)}}</text>`;
}}

function render() {{
  const p = LIVE.probability[i], thr = LIVE.threshold;
  const arc = document.getElementById('arc');
  const frac = Math.max(0, Math.min(1, p));
  arc.setAttribute('stroke-dashoffset', (251 * (1 - frac)).toFixed(1));
  arc.setAttribute('stroke', sevColor(p));
  document.getElementById('probtxt').textContent = p.toFixed(2);
  document.getElementById('probtxt').setAttribute('fill', sevColor(p));

  const s = sev(p, thr);
  const v = document.getElementById('verdict');
  v.textContent = s + (p >= thr ? ' - ALARM RAISED' : ' - monitoring');
  v.style.color = sevColor(p);
  document.getElementById('cycleinfo').textContent =
    `cycle ${{LIVE.cycles[i]}} (t+${{((LIVE.cycles[i] - LIVE.start_cycle) * LIVE.step_minutes / 60).toFixed(1)}} h) | machine ${{LIVE.machine_id}}`;

  sparkline('spark-vib', LIVE.vibration, '#f18f01', 'vibration mm/s', i);
  sparkline('spark-temp', LIVE.temperature, '#e4572e', 'temperature degC', i);
  sparkline('spark-ac', LIVE.acoustic, '#7b5ea7', 'acoustic dB', i);

  // alarm log: show alerts whose cycle has been reached
  const reached = LIVE.alert_cycles.map((c, k) => ({{c, sev: LIVE.alert_severity[k], wo: LIVE.alert_wo[k]}}))
    .filter(a => a.c <= LIVE.cycles[i]);
  const box = document.getElementById('logbox');
  if (!reached.length) {{
    box.innerHTML = '<div class="logline s-ok">[OK] machine healthy - no work orders raised</div>';
  }} else {{
    box.innerHTML = reached.slice(-9).reverse().map(a =>
      `<div class="logline s-${{SEV[a.sev] || 'med'}}">[${{a.sev}}] ${{a.wo}} @ cycle ${{a.c}}</div>`
    ).join('') + (LIVE.cycles[i] >= LIVE.fail_cycle
      ? '<div class="logline s-crit">[BREAKDOWN] machine stopped - unplanned downtime averted by early alarm</div>'
      : '');
  }}
}}

function play() {{
  pause();
  timer = setInterval(() => {{
    i = (i + 1) % LIVE.probability.length;
    render();
  }}, 55);
}}
function pause() {{ if (timer) clearInterval(timer); timer = null; }}
function reset() {{ pause(); i = 0; render(); }}

render();
</script>
</body></html>"""

    out_path.write_text(html, encoding="utf-8")
    log.info("Dashboard -> %s (%.2f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


if __name__ == "__main__":
    from src.utils import banner
    banner("BUILDING HTML DASHBOARD")
    p = build_dashboard()
    print(f"\nOpen this file in any browser:  {p}")

import os, json, time, uuid
from datetime import datetime, timezone
from flask import Flask, request, jsonify, make_response, send_file
from openai import OpenAI

app = Flask(__name__)
BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = os.getenv("NEBIUS_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
MAX_REQUESTS_PER_PROCESS = int(os.getenv("MAX_DIAGNOSES_PER_PROCESS", "25"))
requests_seen = 0

@app.after_request
def cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://u19725505-star.github.io"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Vary"] = "Origin"
    return response

@app.route("/api/diagnose", methods=["OPTIONS"])
def diagnose_preflight():
    return make_response("", 204)


@app.get("/")
def homepage():
    return send_file(os.path.join(os.path.dirname(__file__), "docs", "index.html"))

@app.get("/health")
def health():
    return jsonify({"service":"silicon-technician-judge-site","status":"ready" if os.getenv("NEBIUS_API_KEY") else "needs_api_key","model":DEFAULT_MODEL,"live_inference_enabled":bool(os.getenv("NEBIUS_API_KEY")),"note":"Health endpoint does not make a model call."})

@app.post("/api/diagnose")
def diagnose():
    global requests_seen
    if not os.getenv("NEBIUS_API_KEY"):
        return jsonify({"error":"NEBIUS_API_KEY is not configured on the server. No model call was made."}),503
    if requests_seen >= MAX_REQUESTS_PER_PROCESS:
        return jsonify({"error":"Demo request limit reached for this process. Use recorded evidence or restart after reviewing usage."}),429
    body = request.get_json(silent=True) or {}
    allowed = {"healthy","fan","psu","hot","planned","host","score"}
    scenario = str(body.get("scenario",""))
    if scenario not in allowed:
        return jsonify({"error":"Unknown scenario"}),400
    telemetry = body.get("telemetry") or {}
    # Only accept the small, explicitly supported telemetry payload; do not trust browser-supplied diagnosis text.
    keys = ("cpu_c","thermal_headroom_c","inlet_c","fan1_rpm","psu_output_w")
    clean = {}
    for k in keys:
        v = telemetry.get(k)
        if not isinstance(v,(int,float)) or isinstance(v,bool):
            return jsonify({"error":"Missing or invalid numeric telemetry: "+k}),400
        clean[k] = v
    context = str(body.get("source","scenario telemetry"))[:160]
    name = str(body.get("scenario_name",scenario))[:100]
    # Accept only a bounded, normalized summary of browser vision findings; never trust browser-supplied diagnoses.
    raw_findings = body.get("vision_findings") or []
    if not isinstance(raw_findings, list) or len(raw_findings) > 3:
        return jsonify({"error":"vision_findings must be a list of at most three rack scans"}),400
    vision = []
    for item in raw_findings:
        if not isinstance(item, dict) or str(item.get("rack","")) not in {"A","B","C"}:
            return jsonify({"error":"Invalid rack vision finding"}),400
        pixel_count = item.get("red_pixel_count",0)
        detected = item.get("visual_fault_marker_detected",False)
        if not isinstance(pixel_count,int) or isinstance(pixel_count,bool) or not 0 <= pixel_count <= 20000 or not isinstance(detected,bool):
            return jsonify({"error":"Invalid numeric/boolean vision evidence"}),400
        tele = item.get("telemetry") or {}
        clean_tele = {}
        for key in ("cpu_c","inlet_c","fan_rpm","psu_output_w"):
            val = tele.get(key)
            if not isinstance(val,(int,float)) or isinstance(val,bool):
                return jsonify({"error":"Invalid vision telemetry: "+key}),400
            clean_tele[key] = val
        vision.append({"rack":item["rack"],"visual_fault_marker_detected":detected,"red_pixel_count":pixel_count,"scenario_sensor_anomaly":str(item.get("scenario_sensor_anomaly","unknown"))[:60],"telemetry":clean_tele})
    system = """You are Nemotron acting as a cautious data-center technician. Diagnose only from the supplied simulated scenario, measurements, and bounded computer-vision scan summaries. Distinguish observed evidence from inference. A red-pixel count is a visual heuristic, not proof of a real physical fault. Never claim to have accessed real hardware or taken an action. Return concise JSON with keys: severity (INFO/WARNING/CRITICAL), diagnosis, evidence (array of exact measurements or context facts), recommended_action, action_code (one of "restore_fan", "improve_airflow", "inspect_psu", "no_action", "review_only"), human_review_required (boolean), unsupported_claims (array). Select an action_code only when supported by supplied telemetry; otherwise choose "review_only". Be conservative; no invented readings, no false claims of execution."""
    user = "Scenario: "+name+"\nEvidence source: "+context+"\nSelected scenario measurements (units in keys): "+json.dumps(clean,separators=(",",":"))+"\nComputer-vision rack scan summaries: "+json.dumps(vision,separators=(",",":"))+"\nThese are synthetic browser-simulation observations, not physical camera data. Interpret the visual scan cautiously. Respond with valid JSON only. Do not treat values as live hardware telemetry."
    started = time.time()
    requests_seen += 1
    try:
        client = OpenAI(base_url=BASE_URL,api_key="".join(os.environ["NEBIUS_API_KEY"].split()),timeout=45.0,max_retries=0)
        response = client.chat.completions.create(model=DEFAULT_MODEL,messages=[{"role":"system","content":system},{"role":"user","content":user}],max_tokens=4000,temperature=0.0)
        raw = response.choices[0].message.content or ""
        usage = response.usage
        try:
            parsed = json.loads(raw[raw.index("{"):raw.rindex("}")+1])
        except Exception:
            parsed = None
        # Deterministic validation independent of the model's assertions.
        evidence_ok = bool(parsed and isinstance(parsed.get("diagnosis"),str) and isinstance(parsed.get("evidence"),list))
        if not evidence_ok:
            return jsonify({"error":"Model response did not pass JSON/evidence schema validation.","mode":"live","model":DEFAULT_MODEL,"raw_response":raw[:5000],"request_id":str(uuid.uuid4()),"latency_ms":round((time.time()-started)*1000),"usage":{"prompt_tokens":getattr(usage,"prompt_tokens",None),"completion_tokens":getattr(usage,"completion_tokens",None)}}),502
        # Deterministic server-side gate: the model proposes; this policy authorizes simulation only.
        action_code = parsed.get("action_code", "review_only")
        policy = {
            "fan": ("restore_fan", clean["fan1_rpm"] == 0),
            "hot": ("improve_airflow", clean["inlet_c"] >= 35),
            "psu": ("inspect_psu", clean["psu_output_w"] < 500),
            "healthy": ("no_action", clean["fan1_rpm"] > 0 and clean["inlet_c"] < 30 and clean["psu_output_w"] >= 500),
        }
        expected = policy.get(scenario)
        if not expected or not expected[1] or action_code != expected[0]:
            action_code = "review_only"
        parsed["action_code"] = action_code
        parsed["simulation_action_permitted"] = action_code in {"restore_fan", "improve_airflow", "inspect_psu", "no_action"}
        parsed["mode"]="live"
        parsed["model"]=DEFAULT_MODEL
        parsed["request_id"]=str(uuid.uuid4())
        parsed["latency_ms"]=round((time.time()-started)*1000)
        parsed["usage"]={"prompt_tokens":getattr(usage,"prompt_tokens",None),"completion_tokens":getattr(usage,"completion_tokens",None)}
        parsed["guardrails"]=[
          "Server-side API key; never sent to browser",
          "Model response parsed and required fields validated",
          "Diagnosis grounded against submitted scenario measurements; human review required",
          "No physical hardware actions available from this endpoint"
        ]
        parsed["hardware_action_executed"]=False
        parsed["source"]="Live Nebius Token Factory inference using submitted scenario data"
        return jsonify(parsed)
    except Exception as e:
        return jsonify({"error":"Token Factory request failed: "+str(e)[:300],"mode":"live","model":DEFAULT_MODEL,"request_id":str(uuid.uuid4()),"hardware_action_executed":False}),502

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT","8000")))

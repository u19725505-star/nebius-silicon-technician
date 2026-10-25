import os, json, time, uuid
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = os.getenv("NEBIUS_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b")
MAX_REQUESTS_PER_PROCESS = int(os.getenv("MAX_DIAGNOSES_PER_PROCESS", "25"))
requests_seen = 0

@app.get("/")
def health():
    return jsonify({"service":"silicon-technician-diagnosis","status":"ready" if os.getenv("NEBIUS_API_KEY") else "needs_api_key","model":DEFAULT_MODEL,"live_inference_enabled":bool(os.getenv("NEBIUS_API_KEY")),"note":"Health endpoint does not make a model call."})

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
    system = """You are Nemotron acting as a cautious data-center technician. Diagnose only from the supplied scenario and measurements. Distinguish observed evidence from inference. Never claim to have accessed real hardware or taken an action. Return concise JSON with keys: severity (INFO/WARNING/CRITICAL), diagnosis, evidence (array of exact measurements or context facts), recommended_action, human_review_required (boolean), unsupported_claims (array). Be conservative; no invented readings, no false claims of execution."""
    user = "Scenario: "+name+"\nEvidence source: "+context+"\nMeasurements (units in keys): "+json.dumps(clean,separators=(",",":"))+"\nRespond with valid JSON only. Do not treat the values as live hardware telemetry."
    started = time.time()
    requests_seen += 1
    try:
        client = OpenAI(base_url=BASE_URL,api_key="".join(os.environ["NEBIUS_API_KEY"].split()),timeout=45.0,max_retries=0)
        response = client.chat.completions.create(model=DEFAULT_MODEL,messages=[{"role":"system","content":system},{"role":"user","content":user}],max_tokens=500,temperature=0.0)
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

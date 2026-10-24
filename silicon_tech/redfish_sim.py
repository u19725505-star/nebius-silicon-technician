"""Dynamic Redfish BMC emulator + client (standard library only).
THIS IS AN EMULATOR. Nothing here talks to real hardware. The JSON follows
standard Redfish shapes, so the same client works against a real BMC base URL.

Run:  python -m silicon_tech.redfish_sim          (tour of all scenarios)
      python -m silicon_tech.redfish_sim --serve  (leave running on :8800)
"""
import json
import random
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# last scenario: the BMC looks completely normal; the fault is host-side
SCENARIOS = ["healthy", "fan_failure", "psu_degraded", "inlet_hot",
             "planned_fan_test", "cpu_starvation"]
RANK = {"OK": 0, "Warning": 1, "Critical": 2}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _h(value, warn, crit):
    return "Critical" if value >= crit else ("Warning" if value >= warn else "OK")


class BMCState:
    def __init__(self, seed=1):
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.scenario = "healthy"
        self.t0 = time.time()
        self.led = False
        self.audit = []
        self.log = []
        self.add_log("OK", "BMC started")

    def add_log(self, severity, message):
        self.log.append({"Id": str(len(self.log) + 1),
                         "Created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "Severity": severity, "Message": message})

    def record(self, action, body):
        with self.lock:
            self.audit.append({"time": time.strftime("%H:%M:%S"), "action": action, "body": body})

    def set_scenario(self, name):
        msgs = {
            "fan_failure": ("Critical", "Fan 3 speed below lower critical threshold"),
            "psu_degraded": ("Warning", "Power supply 2 output degraded; redundancy lost"),
            "inlet_hot": ("Warning", "Inlet temperature above upper non-critical threshold"),
            "planned_fan_test": ("OK", "Scheduled fan diagnostic started by maintenance job"),
        }
        with self.lock:
            self.scenario = name
            self.t0 = time.time()
            self.log = []
            self.add_log("OK", "Event log cleared by emulator")
            if name in msgs:
                self.add_log(*msgs[name])

    def readings(self):
        with self.lock:
            sc = self.scenario
            ramp = min(1.0, (time.time() - self.t0) / 4.0)
            g = self.rng.gauss
            fans = [9000 + g(0, 80) for _ in range(6)]
            cpu, gpu, inlet = 58 + g(0, 0.6), 62 + g(0, 0.7), 24 + g(0, 0.2)
            total = 1400 + g(0, 8)
            psu = [(total / 2, "OK"), (total / 2, "OK")]
            if sc == "fan_failure":
                fans[2] = 0.0
                cpu += 30 * ramp
                gpu += 24 * ramp
            elif sc == "psu_degraded":
                p2 = total / 2 * (1 - ramp)
                psu = [(total - p2, "OK"), (p2, "Warning" if ramp <= 0.5 else "Critical")]
            elif sc == "inlet_hot":
                inlet += 14 * ramp
                cpu += 12 * ramp
                gpu += 14 * ramp
                fans = [f + 3500 * ramp for f in fans]
            elif sc == "planned_fan_test":
                fans = [15000 + g(0, 50) for _ in range(6)]
        health = [_h(cpu, 85, 95), _h(gpu, 88, 95), _h(inlet, 35, 42)]
        health += ["Critical" if f < 2000 else "OK" for f in fans]
        health += [h for _, h in psu]
        overall = max(health, key=lambda x: RANK[x])
        return {"cpu": cpu, "gpu": gpu, "inlet": inlet, "fans": fans, "psu": psu, "overall": overall}


class Handler(BaseHTTPRequestHandler):
    state = None

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}

    def do_GET(self):
        s = self.state
        path = self.path.split("?")[0].rstrip("/")
        if path == "/redfish/v1":
            return self._send(200, {"RedfishVersion": "1.15.0",
                                    "Name": "Emulated BMC (not real hardware)"})
        if path == "/emulator/audit":
            return self._send(200, {"actions": s.audit})
        r = s.readings()
        if path == "/redfish/v1/Chassis/1/Thermal":
            temps = [
                {"Name": "CPU Temp", "ReadingCelsius": round(r["cpu"], 1),
                 "UpperThresholdNonCritical": 85, "UpperThresholdCritical": 95,
                 "Status": {"Health": _h(r["cpu"], 85, 95)}},
                {"Name": "GPU Temp", "ReadingCelsius": round(r["gpu"], 1),
                 "UpperThresholdNonCritical": 88, "UpperThresholdCritical": 95,
                 "Status": {"Health": _h(r["gpu"], 88, 95)}},
                {"Name": "Inlet Temp", "ReadingCelsius": round(r["inlet"], 1),
                 "UpperThresholdNonCritical": 35, "UpperThresholdCritical": 42,
                 "Status": {"Health": _h(r["inlet"], 35, 42)}},
            ]
            fans = [{"Name": f"Fan {i + 1}", "Reading": int(v), "ReadingUnits": "RPM",
                     "LowerThresholdCritical": 2000,
                     "Status": {"Health": "Critical" if v < 2000 else "OK", "State": "Enabled"}}
                    for i, v in enumerate(r["fans"])]
            return self._send(200, {"Temperatures": temps, "Fans": fans})
        if path == "/redfish/v1/Chassis/1/Power":
            psus = [{"Name": f"PSU {i + 1}", "PowerOutputWatts": round(w),
                     "Status": {"Health": h, "State": "Enabled"}}
                    for i, (w, h) in enumerate(r["psu"])]
            red = "Warning" if any(h != "OK" for _, h in r["psu"]) else "OK"
            return self._send(200, {
                "PowerControl": [{"PowerConsumedWatts": round(sum(w for w, _ in r["psu"]))}],
                "PowerSupplies": psus,
                "Redundancy": [{"Mode": "Failover", "Status": {"Health": red}}]})
        if path == "/redfish/v1/Systems/1":
            return self._send(200, {"PowerState": "On", "Status": {"Health": r["overall"]},
                                    "LocationIndicatorActive": s.led})
        if path == "/redfish/v1/Managers/1/LogServices/SEL/Entries":
            with s.lock:
                members = list(s.log)[-20:]
            return self._send(200, {"Members": members, "Members@odata.count": len(members)})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        s = self.state
        body = self._body()
        path = self.path.split("?")[0].rstrip("/")
        if path == "/emulator/scenario":
            name = body.get("name")
            if name not in SCENARIOS:
                return self._send(400, {"error": "unknown scenario", "valid": SCENARIOS})
            s.set_scenario(name)
            return self._send(200, {"scenario": name})
        if path == "/redfish/v1/Systems/1/Actions/ComputerSystem.Reset":
            s.record("Reset", body)
            return self._send(200, {"Message": "Reset accepted by emulator; no hardware affected"})
        self._send(404, {"error": "not found"})

    def do_PATCH(self):
        s = self.state
        body = self._body()
        path = self.path.split("?")[0].rstrip("/")
        if path == "/redfish/v1/Systems/1" and "LocationIndicatorActive" in body:
            s.led = bool(body["LocationIndicatorActive"])
            s.record("LocateLED", body)
            return self._send(200, {"LocationIndicatorActive": s.led})
        self._send(404, {"error": "not found"})


# ---------------- client: works against this emulator or a real BMC ----------------
def _req(base, path, body=None, method="GET"):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=5) as r:
        return json.load(r)


def anomalies(s):
    """Deterministic checks done by code, so the model never does arithmetic."""
    out = []
    for n, t in s["temps"].items():
        if t["health"] != "OK":
            out.append(f"{n} {t['c']}C, {t['headroom_c']}C below critical ({t['health']})")
    for n, f in s["fans"].items():
        if f["health"] != "OK":
            out.append(f"{n} at {f['rpm']} RPM ({f['health']})")
        elif f["rpm"] > 13000:
            out.append(f"{n} at {f['rpm']} RPM (unusually high)")
    for n, p in s["psu"].items():
        if p["health"] != "OK":
            out.append(f"{n} output {p['watts']} W ({p['health']})")
    if s["redundancy"] != "OK":
        out.append(f"power redundancy {s['redundancy']}")
    for sev, msg in s["log"]:
        if sev != "OK":
            out.append(f"log {sev}: {msg}")
    return out


def snapshot(base):
    th = _req(base, "/redfish/v1/Chassis/1/Thermal")
    pw = _req(base, "/redfish/v1/Chassis/1/Power")
    sy = _req(base, "/redfish/v1/Systems/1")
    sel = _req(base, "/redfish/v1/Managers/1/LogServices/SEL/Entries")
    snap = {
        "temps": {t["Name"]: {"c": t["ReadingCelsius"],
                              "headroom_c": round(t["UpperThresholdCritical"] - t["ReadingCelsius"], 1),
                              "health": t["Status"]["Health"]} for t in th["Temperatures"]},
        "fans": {f["Name"]: {"rpm": f["Reading"], "health": f["Status"]["Health"]} for f in th["Fans"]},
        "psu": {p["Name"]: {"watts": p["PowerOutputWatts"], "health": p["Status"]["Health"]}
                for p in pw["PowerSupplies"]},
        "redundancy": pw["Redundancy"][0]["Status"]["Health"],
        "consumed_w": pw["PowerControl"][0]["PowerConsumedWatts"],
        "health": sy["Status"]["Health"],
        "log": [(e["Severity"], e["Message"]) for e in sel["Members"][-5:]],
    }
    snap["anomalies"] = anomalies(snap)
    return snap


def _start(port):
    Handler.state = BMCState()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{port}"


def demo(port=8800):
    srv, base = _start(port)
    for name in SCENARIOS:
        _req(base, "/emulator/scenario", {"name": name}, "POST")
        time.sleep(5)
        s = snapshot(base)
        print(f"\n== {name}   overall={s['health']}   consumed={s['consumed_w']} W")
        for a in s["anomalies"] or ["none"]:
            print("   -", a)
    print("\nsafe action test: light the locate LED")
    print("  ", _req(base, "/redfish/v1/Systems/1", {"LocationIndicatorActive": True}, "PATCH"))
    print("   audit:", _req(base, "/emulator/audit")["actions"])
    srv.shutdown()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        srv, base = _start(8800)
        print(f"Emulated BMC on {base}/redfish/v1  (Ctrl-C to stop)")
        threading.Event().wait()
    else:
        demo()
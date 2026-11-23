"""HTTP-layer tests. They need fastapi + httpx and are SKIPPED when those are not installed.
(They were not runnable in the build sandbox; run `pip install -e .[dev]` and `pytest` to execute them.)"""
import json
import unittest

try:
    from fastapi.testclient import TestClient
    HAVE = True
except Exception:  # pragma: no cover
    HAVE = False

if HAVE:
    from switchboard.api.app import create_app
    from tests.helpers import mk_gateway


@unittest.skipUnless(HAVE, "fastapi/httpx not installed")
class TestAPI(unittest.TestCase):
    def setUp(self):
        self.gw = mk_gateway()
        self.cm = TestClient(create_app(gateway=self.gw))
        self.c = self.cm.__enter__()

    def tearDown(self):
        self.cm.__exit__(None, None, None)

    def body(self, **kw):
        return {"model": "gpt-4o", "temperature": 0, "messages": [{"role": "user", "content": "hello there"}], **kw}

    def test_chat_schema_and_headers(self):
        r = self.c.post("/v1/chat/completions", json=self.body())
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["object"], "chat.completion")
        self.assertIn("usage", d)
        self.assertEqual(r.headers["x-cache"], "MISS")
        self.assertEqual(self.c.post("/v1/chat/completions", json=self.body()).headers["x-cache"], "HIT-EXACT")

    def test_sse_stream_format(self):
        r = self.c.post("/v1/chat/completions", json=self.body(stream=True, temperature=1.0))
        lines = [l for l in r.text.split("\n\n") if l]
        self.assertTrue(lines[-1] == "data: [DONE]")
        first = json.loads(lines[0][6:])
        self.assertEqual(first["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(json.loads(lines[-2][6:])["choices"][0]["finish_reason"], "stop")

    def test_errors_are_openai_shaped(self):
        r = self.c.post("/v1/chat/completions", json={"model": "x"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.json())
        self.assertEqual(self.c.post("/v1/chat/completions", json=self.body(model="nope-1")).status_code, 404)

    def test_models_health_metrics(self):
        self.assertIn("auto", [m["id"] for m in self.c.get("/v1/models").json()["data"]])
        self.assertEqual(self.c.get("/health").json()["status"], "ok")
        self.assertIn("switchboard_", self.c.get("/metrics").text)

    def test_jobs_roundtrip(self):
        r = self.c.post("/v1/jobs", json={"requests": [self.body(model="mock-balanced")] * 3, "idempotency_key": "k"})
        self.assertEqual(r.status_code, 202)
        jid = r.json()["id"]
        self.assertEqual(self.c.post("/v1/jobs", json={"requests": [self.body()] * 3, "idempotency_key": "k"}).json()["id"], jid)
        self.assertEqual(self.c.get(f"/v1/jobs/{jid}").json()["total"], 3)
        self.assertEqual(self.c.post(f"/v1/jobs/{jid}/cancel").status_code, 200)

    def test_admin_and_auth(self):
        self.gw.settings.admin_key = "adm"
        self.assertEqual(self.c.get("/admin/savings").status_code, 403)
        self.assertEqual(self.c.get("/admin/savings", headers={"X-Admin-Key": "adm"}).status_code, 200)
        self.gw.settings.require_auth = True
        self.assertEqual(self.c.post("/v1/chat/completions", json=self.body()).status_code, 401)


if __name__ == "__main__":
    unittest.main()

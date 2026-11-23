"""Real process-level crash test: SIGKILL a worker mid-job, start another, verify exactly-once."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from switchboard.clock import Clock
from switchboard.db.session import Database
from switchboard.scheduler.queue import TaskQueue

ROOT = Path(__file__).resolve().parents[2]


class TestCrash(unittest.TestCase):
    def test_kill9_worker_then_recover_no_loss_no_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            dbp = os.path.join(d, "sb.db")
            db = Database(dbp)
            q = TaskQueue(db, Clock())
            n = 60
            jid, _ = q.create_job("anonymous", [{"model": "mock-balanced", "temperature": 0,
                                                 "messages": [{"role": "user", "content": f"crash test {i}"}]}
                                                for i in range(n)])
            env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "MOCK_LATENCY_SCALE": "15", "ENABLE_MOCK": "true"}
            cmd = [sys.executable, "-m", "switchboard", "worker", "--db", dbp, "--concurrency", "6",
                   "--lease-seconds", "2", "--until-idle"]
            p1 = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            deadline = time.time() + 20
            while time.time() < deadline and not (0 < q.counts(jid)["done"] < n and q.counts(jid)["leased"] > 0):
                time.sleep(0.05)
            c_before = q.counts(jid)
            self.assertTrue(0 < c_before["done"] < n, c_before)
            self.assertGreater(c_before["leased"], 0, "need in-flight tasks at the moment of the crash")
            os.kill(p1.pid, signal.SIGKILL)
            p1.wait()
            self.assertGreater(q.counts(jid)["leased"], 0)  # orphaned leases
            p2 = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.assertEqual(p2.wait(timeout=90), 0)
            c = q.counts(jid)
            self.assertEqual((c["done"], c["dead"], c["pending"], c["leased"]), (n, 0, 0, 0))
            rows = db.query("SELECT task_key, COUNT(*) n FROM usage WHERE job_id=? AND outcome='ok' GROUP BY task_key", (jid,))
            self.assertEqual(len(rows), n)
            self.assertTrue(all(r["n"] == 1 for r in rows))  # charged exactly once
            self.assertEqual(len(list(q.results(jid))), n)
            reclaimed = db.query("SELECT COUNT(*) n FROM tasks WHERE job_id=? AND attempts>1", (jid,))[0]["n"]
            self.assertGreater(reclaimed, 0)


if __name__ == "__main__":
    unittest.main()

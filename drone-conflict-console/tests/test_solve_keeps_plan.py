"""Route-level guard for an *unverified* re-solve (status unknown).

Uses a temporary SQLite store and monkeypatches ``app.solver.run_solve`` so
that a first (real, tiny) solve is followed by a re-solve whose probe timed
out (``status == "unknown"``).  The rules pinned here:

* the unknown response stays ``unknown`` in ``solution`` (no false success);
* the response carries THIS attempt's locks/thresholds (they drove the
  conflict constraints and the solver), and the old verified table separately
  in ``prior_solution`` together with its ORIGINAL thresholds and constraint
  table, so the page can still view the old table without mistaking it for a
  solution under the new constraints;
* nothing is written to SQLite, so a later GET still serves the old table;
* without an old table an unknown response has ``prior_solution == null``;
* creating a batch from the retained table must be explicit
  (``plan_source="stored"``) and echo the old thresholds — a mismatch or a
  missing echo is rejected; the default request shape stays compatible;
* a genuine ``infeasible`` proof still replaces the stored result, and a new
  feasible witness is saved as before.

Run:  python tests/test_solve_keeps_plan.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app import main as web  # noqa: E402
from app.solver import SolveResult  # noqa: E402

FAILS = 0


def check(cond, msg):
    global FAILS
    if not cond:
        FAILS += 1
        print("FAIL:", msg)
    else:
        print("ok:", msg)


CROSSROADS = {
    "safety_radius": 30,
    "safety_height": 10,
    "sorties": [
        {"id": "A", "weight": 1, "window_hi": 20,
         "waypoints": [
             {"x": 0, "y": 0, "z": 20, "t": 0},
             {"x": 200, "y": 0, "z": 20, "t": 20}]},
        {"id": "B", "weight": 1, "window_hi": 20,
         "waypoints": [
             {"x": 100, "y": -100, "z": 20, "t": 0},
             {"x": 100, "y": 100, "z": 20, "t": 20}]},
    ],
}


def main():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "console.db")
    web.DB_PATH = db_path
    web.store = web.Store(db_path)

    with TestClient(web.app) as client:
        r = client.post("/api/import", json=CROSSROADS)
        check(r.status_code == 200, f"import ok ({r.status_code})")
        sid = r.json()["scenario_id"]

        # First solve: real solver, known feasible plan persisted at R=30/H=10.
        r = client.post(f"/api/scenarios/{sid}/solve", json={"time_limit": 20})
        check(r.status_code == 200, f"first solve ok ({r.status_code})")
        first = r.json()["solution"]
        check(first["feasible"], "first solve is feasible")
        check("prior_solution" not in r.json(),
              "successful solve keeps its original response shape (no prior_solution)")
        saved_offsets = dict(first["offsets"])

        # Re-solve whose probe times out, under NEW thresholds + a NEW lock.
        unknown = SolveResult(feasible=False, status="unknown",
                              locked={"A": 7})
        original = web.run_solve

        def fake_unknown(*args, **kwargs):
            return unknown

        web.run_solve = fake_unknown
        try:
            r = client.post(f"/api/scenarios/{sid}/solve", json={
                "time_limit": 20,
                "safety_radius": 55,
                "safety_height": 22,
                "locks": [{"sortie_id": "A", "offset": 7}],
            })
        finally:
            web.run_solve = original
        check(r.status_code == 200, f"unknown solve still 200 ({r.status_code})")
        body = r.json()

        # The response must say unknown — never dress the old table up as this
        # solve's solution.
        sol = body["solution"]
        check(sol["status"] == "unknown" and not sol["feasible"],
              "response solution stays unknown on timed-out re-solve")
        check(sol["locked"] == {"A": 7}, "unknown carries this attempt's locks")
        check(body["safety_radius"] == 55 and body["safety_height"] == 22,
              "response carries this attempt's thresholds (not the old ones)")

        # The old verified table is carried separately, tagged with the
        # ORIGINAL thresholds and its own constraint table.
        prior = body.get("prior_solution")
        check(prior is not None, "prior_solution carries the retained old table")
        if prior:
            check(prior["result"]["feasible"]
                  and prior["result"]["offsets"] == saved_offsets,
                  "prior_solution holds the previously verified table")
            check(prior["safety_radius"] == 30 and prior["safety_height"] == 10,
                  "prior_solution states the OLD table's original thresholds")
            check(isinstance(prior.get("pair_constraints"), list),
                  "prior_solution includes the old table's constraint table")

        # SQLite was never touched: GET still serves the old plan and thresholds.
        r = client.get(f"/api/scenarios/{sid}")
        dto = r.json()
        persisted = dto["solution"]
        check(persisted["feasible"] and persisted["offsets"] == saved_offsets,
              "unknown re-solve never overwrote the stored plan")
        check(dto["safety_radius"] == 30 and dto["safety_height"] == 10,
              "stored thresholds remain the old plan's thresholds")
        check("prior_solution" not in dto,
              "GET scenario keeps its original response shape (no prior_solution)")

        # The normal batch entry must not silently freeze the old constraints.
        r = client.post(f"/api/scenarios/{sid}/batches", json={"name": "naive"})
        check(r.status_code == 201,
              f"default request still works, frozen on stored row ({r.status_code})")
        check(r.json()["safety_radius"] == 30
              and r.json()["planned_offsets"] == saved_offsets,
              "default-compatible batch freezes the stored verified plan")

        # Explicit old-table selection requires the original threshold echo.
        r = client.post(f"/api/scenarios/{sid}/batches", json={
            "name": "stored-noecho", "plan_source": "stored"})
        check(r.status_code == 400,
              f"stored without echoed thresholds rejected ({r.status_code})")

        r = client.post(f"/api/scenarios/{sid}/batches", json={
            "name": "stored-wrong",
            "plan_source": "stored",
            "stored_safety_radius": 55,
            "stored_safety_height": 22})
        check(r.status_code == 400,
              f"stored with wrong (new) thresholds rejected ({r.status_code})")
        check(r.json()["detail"]["reason"] == "stored_threshold_mismatch",
              "mismatch explains the stored vs submitted thresholds")

        r = client.post(f"/api/scenarios/{sid}/batches", json={
            "name": "stored-ok",
            "plan_source": "stored",
            "stored_safety_radius": 30,
            "stored_safety_height": 10})
        check(r.status_code == 201,
              f"explicit stored selection with original thresholds works ({r.status_code})")
        check(r.json()["planned_offsets"] == saved_offsets
              and r.json()["safety_radius"] == 30
              and r.json()["safety_height"] == 10,
              "explicit old-table batch freezes the retained plan at old thresholds")

        # Unknown with NO prior table: unknown, and no retained table offered.
        r = client.post("/api/import", json=CROSSROADS)
        sid2 = r.json()["scenario_id"]
        web.run_solve = fake_unknown
        try:
            r = client.post(f"/api/scenarios/{sid2}/solve", json={"time_limit": 20})
        finally:
            web.run_solve = original
        check(r.status_code == 200 and r.json()["solution"]["status"] == "unknown",
              "first-time unknown stays unknown")
        check(r.json()["prior_solution"] is None,
              "no old table -> prior_solution is null")
        r = client.get(f"/api/scenarios/{sid2}")
        check(r.json().get("solution") is None,
              "unknown with no prior table does not persist anything")
        r = client.post(f"/api/scenarios/{sid2}/batches", json={"name": "x"})
        check(r.status_code == 400, "no batch without a verified feasible plan")

        # Back on scenario 1: a genuine infeasibility proof still replaces the
        # stored result (distinct from unknown).
        infeas = SolveResult(feasible=False, status="infeasible",
                             cores=[{"sorties": ["A", "B"], "size": 2,
                                     "mutex_offsets": []}],
                             truncated=False)
        web.run_solve = lambda *a, **k: infeas  # noqa: E731
        r = client.post(f"/api/scenarios/{sid}/solve", json={"time_limit": 20})
        web.run_solve = original
        check(r.status_code == 200 and not r.json()["solution"]["feasible"],
              "proven infeasible replaces the stored result (distinct from unknown)")
        check("prior_solution" not in r.json(),
              "infeasible response keeps its original shape (no prior_solution)")
        r = client.get(f"/api/scenarios/{sid}")
        check(r.json()["solution"]["status"] == "infeasible",
              "infeasible proof persisted, so 'no solution' stays visible")

        # A new feasible witness after that is saved normally.
        r = client.post(f"/api/scenarios/{sid}/solve", json={"time_limit": 20})
        check(r.status_code == 200 and r.json()["solution"]["feasible"],
              "new feasible solve after infeasible is saved and returned")
        check("prior_solution" not in r.json(),
              "new feasible solve keeps its original response shape")
        r = client.get(f"/api/scenarios/{sid}")
        check(r.json()["solution"]["feasible"],
              "new feasible solution persisted as usual")

    print(f"route guard checks, failures: {FAILS}")
    return FAILS


if __name__ == "__main__":
    sys.exit(1 if main() else 0)

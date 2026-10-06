"""Route-level guard: a timed-out solve must not overwrite a known plan —
and must not *report* success either.

Uses a temporary SQLite store and monkeypatches ``app.solver.run_solve`` so
that a first (real, tiny) solve is followed by a re-solve whose probe timed
out (``status == "unknown"``).  The new contract:

* the response's ``solution`` honestly reports THIS run as ``unknown``
  (no back-filled old table masquerading as a fresh success);
* the response keeps the requested (new) thresholds/constraints at top
  level — they were what the solver actually saw;
* the retained old table rides in ``prior_solution``, annotated with the
  original thresholds it was solved under, so it stays viewable/usable;
* SQLite still holds the old plan and batches can still be (explicitly)
  created from it with its original thresholds;
* with no prior feasible plan the response is a plain ``unknown``;
* a genuine infeasibility proof still replaces the stored result, and a
  later feasible solve saves as usual.

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

        # First solve: real solver, known feasible plan persisted.
        r = client.post(f"/api/scenarios/{sid}/solve", json={"time_limit": 20})
        check(r.status_code == 200, f"first solve ok ({r.status_code})")
        first = r.json()["solution"]
        check(first["feasible"], "first solve is feasible")
        check("prior_solution" not in r.json(),
              "plain successful solve carries no prior_solution field")
        saved_offsets = dict(first["offsets"])

        # Re-solve with NEW thresholds and a NEW lock whose probe times out:
        # nothing proven, nothing known — for the NEW constraints.
        original = web.run_solve
        seen = {}

        def fake_unknown(sorties, pcs, locks, **kwargs):
            seen["locks"] = dict(locks)
            seen["pcs"] = pcs
            # Mirror the real solver's unknown return (it echoes the locks).
            return SolveResult(feasible=False, status="unknown",
                               locked=dict(locks))

        web.run_solve = fake_unknown
        try:
            r = client.post(
                f"/api/scenarios/{sid}/solve",
                json={"time_limit": 20, "safety_radius": 45,
                      "safety_height": 12,
                      "locks": [{"sortie_id": "A", "offset": 3}]},
            )
        finally:
            web.run_solve = original
        check(r.status_code == 200, f"unknown solve still 200 ({r.status_code})")
        body = r.json()
        sol = body["solution"]
        check(sol["status"] == "unknown" and not sol["feasible"],
              "response solution honestly reports THIS run as unknown")
        check(seen.get("locks") == {"A": 3},
              "new locks were forwarded into the solver")
        check(sol["locked"] == {"A": 3},
              "response solution echoes this run's locks")
        check(body["safety_radius"] == 45 and body["safety_height"] == 12,
              "top-level thresholds stay the requested new ones (not reverted)")
        # The retained old table is carried separately, with its ORIGINAL
        # thresholds, so it stays viewable without masking the unverified run.
        prior = body.get("prior_solution")
        check(prior is not None, "retained old table exposed as prior_solution")
        check(prior and prior["feasible"] and prior["offsets"] == saved_offsets,
              "prior_solution carries the retained old offsets")
        check(prior and prior["safety_radius"] == 30
              and prior["safety_height"] == 10,
              "prior_solution is annotated with the old table's original "
              "thresholds")

        # SQLite still holds the old plan under the old thresholds.
        r = client.get(f"/api/scenarios/{sid}")
        got = r.json()
        persisted = got["solution"]
        check(persisted["feasible"] and persisted["offsets"] == saved_offsets,
              "unknown re-solve never overwrote the stored plan")
        check(got["safety_radius"] == 30 and got["safety_height"] == 10,
              "stored thresholds unchanged by the unknown re-solve")
        check("prior_solution" not in got,
              "GET returns the stored plan directly (no prior_solution)")

        # Batches keep being created from the retained plan (compatible), and
        # the explicit old-table selection flag is accepted too.
        r = client.post(f"/api/scenarios/{sid}/batches", json={"name": "kept"})
        check(r.status_code == 201, f"batch created from retained plan ({r.status_code})")
        check(r.json()["planned_offsets"] == saved_offsets,
              "batch frozen on the retained (non-overwritten) plan")
        check(r.json()["safety_radius"] == 30 and r.json()["safety_height"] == 10,
              "batch frozen on the old table's original thresholds")
        r = client.post(f"/api/scenarios/{sid}/batches",
                        json={"name": "kept-explicit", "use_stored_plan": True})
        check(r.status_code == 201
              and r.json()["planned_offsets"] == saved_offsets,
              "explicit use_stored_plan selection freezes the same old table")

        # A fresh feasible solve still saves as usual (and clears the stale
        # situation): the stored result becomes the new table.
        r = client.post(f"/api/scenarios/{sid}/solve", json={"time_limit": 20})
        check(r.status_code == 200 and r.json()["solution"]["feasible"],
              "new feasible solve after unknown saves as usual")
        check("prior_solution" not in r.json(),
              "feasible re-solve carries no prior_solution")

        # A genuine infeasibility proof still replaces the stored result.
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
              "infeasible run carries no prior_solution")
        r = client.get(f"/api/scenarios/{sid}")
        check(r.json()["solution"]["status"] == "infeasible",
              "infeasible proof persisted, so 'no solution' stays visible")

        # No prior feasible plan at all: a bare unknown, nothing to fall back
        # to — the response is a plain unknown without prior_solution.
        r = client.post("/api/import", json=CROSSROADS)
        sid2 = r.json()["scenario_id"]
        web.run_solve = fake_unknown
        try:
            r = client.post(f"/api/scenarios/{sid2}/solve",
                            json={"time_limit": 20})
        finally:
            web.run_solve = original
        body2 = r.json()
        check(body2["solution"]["status"] == "unknown"
              and not body2["solution"]["feasible"],
              "unknown with no prior plan reports unknown")
        check(body2.get("prior_solution") is None,
              "no prior feasible plan -> no prior_solution field")
        r = client.post(f"/api/scenarios/{sid2}/batches", json={"name": "nope"})
        check(r.status_code == 400,
              "no feasible plan stored -> batch creation still rejected")

    print(f"route guard checks, failures: {FAILS}")
    return FAILS


if __name__ == "__main__":
    sys.exit(1 if main() else 0)

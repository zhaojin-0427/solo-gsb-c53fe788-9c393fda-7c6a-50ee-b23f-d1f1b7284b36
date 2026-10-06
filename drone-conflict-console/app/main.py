"""FastAPI application: import, analyse, solve and persist drone scenarios."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .model import Waypoint, make_sortie
from .schemas import (
    CreateBatchRequest,
    ImportPayload,
    LockEntry,
    ReportRequest,
    SaveRequest,
    SolveRequest,
)
from .solver import (
    PairConstraint,
    build_pair_constraints,
    run_solve,
    scan_schedule,
)
from .batch import EarlyTimeReject, Report, evaluate_warning, review_warning
from .workzone import InvalidWorkZoneQuery, evaluate_workzone
from .storage import Store

DB_PATH = os.environ.get("DCC_DB_PATH", "data/console.db")
store: Store | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    store = Store(DB_PATH)
    yield


app = FastAPI(title="无人机航迹冲突消解台", version="1.0", lifespan=lifespan)


def _build_sorties(payload: ImportPayload | dict):
    if isinstance(payload, ImportPayload):
        data = payload
    else:
        data = ImportPayload.model_validate(payload)
    sorties = []
    for s in data.sorties:
        wps = [Waypoint(w.x, w.y, w.z, w.t) for w in s.waypoints]
        sorties.append(make_sortie(s.id, s.weight, s.window_hi, wps, 0))
    return sorties, data.safety_radius, data.safety_height


def _pair_constraints_dto(pcs: list[PairConstraint]) -> list[dict]:
    return [
        {"sortie_a": pc.sortie_a, "sortie_b": pc.sortie_b,
         "forbidden_offsets": [[lo, hi] for lo, hi in pc.intervals]}
        for pc in pcs
    ]


def _scenario_summary(sid: int, sorties, safety_radius, safety_height,
                      pcs, solve_result=None, earliest_conflicts=None,
                      prior_solution="UNSET") -> dict:
    out = {
        "scenario_id": sid,
        "safety_radius": safety_radius,
        "safety_height": safety_height,
        "sorties": [
            {
                "id": s.id,
                "weight": s.weight,
                "window_lo": s.window_lo,
                "window_hi": s.window_hi,
                "flight_end": s.flight_end,
                "waypoints": [
                    {"x": w.x, "y": w.y, "z": w.z, "t": w.t} for w in s.waypoints
                ],
            }
            for s in sorties
        ],
        "pair_constraints": _pair_constraints_dto(pcs),
        "earliest_conflicts": earliest_conflicts,
    }
    if solve_result is not None:
        out["solution"] = solve_result
    # Only the solve route supplies this argument; GET/import responses keep
    # their exact original shape for compatibility.
    if prior_solution != "UNSET":
        out["prior_solution"] = prior_solution
    return out


@app.post("/api/import")
def import_scenario(payload: ImportPayload):
    sorties, R, H = _build_sorties(payload)
    sid = store.create_scenario(payload.model_dump(), R, H)
    pcs = build_pair_constraints(sorties, R, H)
    earliest = scan_schedule(sorties, {s.id: 0 for s in sorties}, R, H)
    return _scenario_summary(sid, sorties, R, H, pcs, earliest_conflicts=earliest)


@app.get("/api/scenarios")
def list_scenarios():
    return {"scenarios": store.list_scenarios()}


@app.get("/api/scenarios/{sid}")
def get_scenario(sid: int):
    row = store.get_scenario(sid)
    if row is None:
        raise HTTPException(404, "scenario not found")
    payload = ImportPayload.model_validate(row["input"])
    sorties, _, _ = _build_sorties(payload)
    R = row["safety_radius"]
    H = row["safety_height"]
    pcs = build_pair_constraints(sorties, R, H)
    earliest = scan_schedule(sorties, {s.id: 0 for s in sorties}, R, H)
    return _scenario_summary(sid, sorties, R, H, pcs,
                             solve_result=row["result"],
                             earliest_conflicts=earliest)


@app.delete("/api/scenarios/{sid}")
def delete_scenario(sid: int):
    if not store.delete_scenario(sid):
        raise HTTPException(404, "scenario not found")
    return {"deleted": sid}


def _load(sid: int):
    row = store.get_scenario(sid)
    if row is None:
        raise HTTPException(404, "scenario not found")
    payload = ImportPayload.model_validate(row["input"])
    sorties, _, _ = _build_sorties(payload)
    return sorties, row, payload


@app.post("/api/scenarios/{sid}/solve")
def solve_scenario(sid: int, req: SolveRequest):
    sorties, row, payload = _load(sid)
    R = req.safety_radius if req.safety_radius is not None else row["safety_radius"]
    H = req.safety_height if req.safety_height is not None else row["safety_height"]
    if R <= 0 or H <= 0:
        raise HTTPException(400, "safety parameters must be positive")
    known = {s.id: s for s in sorties}
    locks: dict[str, int] = {}
    for lk in req.locks:
        if lk.sortie_id not in known:
            raise HTTPException(400, f"unknown sortie id in locks: {lk.sortie_id}")
        s = known[lk.sortie_id]
        if not (s.window_lo <= lk.offset <= s.window_hi):
            raise HTTPException(
                400,
                f"locked offset {lk.offset} for {lk.sortie_id} outside window "
                f"[{s.window_lo}, {s.window_hi}]",
            )
        locks[lk.sortie_id] = lk.offset

    # The just-attempted locks/thresholds always drive the conflict table and
    # the solver, and are echoed back whether or not the proof finishes.
    pcs = build_pair_constraints(sorties, R, H)
    res = run_solve(sorties, pcs, locks, time_limit=req.time_limit)

    prior = row.get("result")
    prior_feasible = bool(prior and prior.get("feasible"))
    prior_block = None

    earliest = scan_schedule(sorties, {s.id: 0 for s in sorties}, R, H)
    solved_conflicts = []
    if res.feasible:
        solved_conflicts = scan_schedule(sorties, res.offsets, R, H)
        if solved_conflicts:
            raise HTTPException(500, "internal error: solved schedule still conflicts")

    if res.status == "unknown":
        # The re-solve proved NOTHING under the new constraints, so the stored
        # row is neither overwritten nor quietly served as "this" solution: the
        # response stays unknown (solution) while the previously verified table
        # is exposed separately, still viewable, with the thresholds under
        # which it was actually proven.  With no prior table, unknown is simply
        # visible with no retained plan.
        if prior_feasible:
            prior_pcs = build_pair_constraints(
                sorties, row["safety_radius"], row["safety_height"])
            prior_block = {
                "result": prior,
                "safety_radius": row["safety_radius"],
                "safety_height": row["safety_height"],
                "pair_constraints": _pair_constraints_dto(prior_pcs),
            }
    else:
        # A proven infeasibility or a new feasible witness replaces the stored
        # result exactly as before (feasible/infeasible rows are always saved).
        store.save_result(sid, _result_dict(res, solved_conflicts), R, H)

    result = _result_dict(res, solved_conflicts)
    # prior_solution is part of the response ONLY for an unknown re-solve;
    # ordinary successful/failed solves keep their exact historical fields.
    prior_arg = prior_block if res.status == "unknown" else "UNSET"
    return _scenario_summary(sid, sorties, R, H, pcs,
                             solve_result=result,
                             earliest_conflicts=earliest,
                             prior_solution=prior_arg)


def _result_dict(res, solved_conflicts: list | None = None) -> dict:
    return {
        "feasible": res.feasible,
        "status": res.status,
        "weighted_delay": res.weighted_delay,
        "offsets": res.offsets,
        "locked": res.locked,
        "elapsed_seconds": round(res.elapsed_seconds, 4),
        "solved_conflicts": solved_conflicts or [],
        "cores": res.cores,
        "truncated": res.truncated,
    }


@app.post("/api/scenarios/{sid}/save")
def save_scenario(sid: int, req: SaveRequest):
    sorties, row, payload = _load(sid)
    if row["result"] is None:
        raise HTTPException(400, "solve the scenario before saving a plan")
    store.save_result(sid, row["result"], row["safety_radius"], row["safety_height"],
                      name=req.name)
    return {"scenario_id": sid, "name": req.name, "saved": True}


# ---------------------------------------------------------------------------
# Execution batches: frozen plan + actual-take-off reports + warning
# ---------------------------------------------------------------------------


def _build_batch(row: dict):
    payload = ImportPayload.model_validate(row["input"])
    sorties, _, _ = _build_sorties(payload)
    reports = {
        r["sortie_id"]: Report(r["actual_offset"], r["revision"])
        for r in row["reports"]
    }
    return sorties, reports


def _batch_summary(row: dict) -> dict:
    return {
        "batch_id": row["id"],
        "scenario_id": row["scenario_id"],
        "name": row["name"],
        "safety_radius": row["safety_radius"],
        "safety_height": row["safety_height"],
        "created_at": row["created_at"],
        "planned_offsets": row["plan"]["offsets"],
        "reports": [
            {"sortie_id": r["sortie_id"], "actual_offset": r["actual_offset"],
             "revision": r["revision"], "reported_at": r["reported_at"]}
            for r in row["reports"]
        ],
    }


@app.post("/api/scenarios/{sid}/batches", status_code=201)
def create_batch(sid: int, req: CreateBatchRequest):
    row = store.get_scenario(sid)
    if row is None:
        raise HTTPException(404, "scenario not found")
    if req.plan_source not in ("current", "stored"):
        raise HTTPException(
            400, "plan_source must be 'current' or 'stored'")
    if req.plan_source == "stored":
        # Explicitly freeze the PREVIOUSLY VERIFIED table that survived an
        # unverified (unknown) re-solve.  The client must echo the old table's
        # original thresholds, so a batch can never silently freeze a plan
        # under constraints the caller mistook for the just-attempted ones.
        result = row["result"]
        if result is None or not result.get("feasible"):
            raise HTTPException(
                400, "no previously verified feasible plan is stored; "
                     "a feasible plan must be solved before creating an execution batch")
        if req.stored_safety_radius is None or req.stored_safety_height is None:
            raise HTTPException(
                400,
                "plan_source=stored requires stored_safety_radius and "
                "stored_safety_height echoing the old plan's original thresholds")
        if (float(req.stored_safety_radius) != float(row["safety_radius"])
                or float(req.stored_safety_height) != float(row["safety_height"])):
            raise HTTPException(400, detail={
                "reason": "stored_threshold_mismatch",
                "submitted_safety_radius": req.stored_safety_radius,
                "submitted_safety_height": req.stored_safety_height,
                "stored_safety_radius": row["safety_radius"],
                "stored_safety_height": row["safety_height"],
            })
        plan_R, plan_H = row["safety_radius"], row["safety_height"]
    else:
        # Default behaviour, unchanged for existing clients: freeze the plan of
        # the most recent verified solve held on the scenario row.
        result = row["result"]
        if result is None or not result.get("feasible"):
            raise HTTPException(
                400, "a feasible plan must be solved before creating an execution batch")
        plan_R, plan_H = row["safety_radius"], row["safety_height"]
    # Freeze the input snapshot, safety thresholds and planned offsets; later
    # re-solves of the scenario never mutate this batch.
    plan = {"offsets": dict(result["offsets"])}
    bid = store.create_batch(sid, req.name, row["input"], plan_R, plan_H, plan)
    batch = store.get_batch(bid)
    return _batch_summary(batch)


@app.get("/api/batches")
def list_batches(scenario_id: int | None = None):
    return {"batches": store.list_batches(scenario_id)}


@app.get("/api/batches/{bid}")
def get_batch(bid: int):
    row = store.get_batch(bid)
    if row is None:
        raise HTTPException(404, "batch not found")
    return _batch_summary(row)


@app.delete("/api/batches/{bid}")
def delete_batch(bid: int):
    if not store.delete_batch(bid):
        raise HTTPException(404, "batch not found")
    return {"deleted": bid}


@app.put("/api/batches/{bid}/reports")
def put_report(bid: int, req: ReportRequest):
    row = store.get_batch(bid)
    if row is None:
        raise HTTPException(404, "batch not found")
    known = {s["id"] for s in row["input"]["sorties"]}
    if req.sortie_id not in known:
        raise HTTPException(404, f"sortie not found in batch: {req.sortie_id}")
    # Atomic compare-and-set: identical retries are idempotent, same revision
    # with a different value or an older revision conflicts, and concurrent
    # reports leave only the highest revision stored.
    outcome, info = store.submit_report(
        bid, req.sortie_id, req.actual_offset, req.revision)
    if outcome in ("idempotent", "inserted"):
        return {"batch_id": bid, "report": info, "accepted": True,
                "idempotent": outcome == "idempotent"}
    if outcome == "invalid_first_revision":
        raise HTTPException(409, detail={
            "reason": "invalid_first_revision",
            "sortie_id": req.sortie_id,
            "received_revision": req.revision,
            "expected_revision": 1,
        })
    if outcome == "revision_value_conflict":
        raise HTTPException(409, detail={
            "reason": "revision_value_conflict",
            "sortie_id": req.sortie_id,
            "revision": req.revision,
            "stored_actual_offset": info["stored_actual_offset"],
            "received_actual_offset": req.actual_offset,
        })
    raise HTTPException(409, detail={
        "reason": "stale_revision",
        "sortie_id": req.sortie_id,
        "received_revision": req.revision,
        "current_revision": info["current_revision"],
    })


@app.get("/api/batches/{bid}/warning")
def batch_warning(bid: int, t: int):
    row = store.get_batch(bid)
    if row is None:
        raise HTTPException(404, "batch not found")
    if t < 0:
        raise HTTPException(400, "t must be a non-negative integer second")
    sorties, reports = _build_batch(row)
    try:
        result = evaluate_warning(
            sorties, row["plan"]["offsets"], reports,
            row["safety_radius"], row["safety_height"], t,
        )
    except EarlyTimeReject as exc:
        raise HTTPException(
            400,
            detail={
                "reason": "t_before_reported_takeoff",
                "t": t,
                "violations": exc.violations,
            },
        )
    summary = _batch_summary(row)
    return {**summary, "warning": result}


@app.get("/api/batches/{bid}/workzone")
def batch_workzone(bid: int, cx: float, cy: float, radius: float,
                   z_lo: float, z_hi: float, t: int, T: int, d: int):
    """Safe integer start window for a temporary cylindrical work zone."""
    row = store.get_batch(bid)
    if row is None:
        raise HTTPException(404, "batch not found")
    sorties, reports = _build_batch(row)
    try:
        result = evaluate_workzone(
            sorties, row["plan"]["offsets"], reports,
            cx, cy, radius, z_lo, z_hi, t, T, d)
    except InvalidWorkZoneQuery as exc:
        raise HTTPException(400, detail={"reason": exc.reason, **exc.fields})
    except EarlyTimeReject as exc:
        raise HTTPException(
            400,
            detail={
                "reason": "t_before_reported_takeoff",
                "t": t,
                "violations": exc.violations,
            },
        )
    summary = _batch_summary(row)
    return {**summary, "workzone": result}


def _reports_from_snapshot(snapshot: list[dict]) -> dict[str, Report]:
    return {
        item["sortie_id"]: Report(item["actual_offset"], item["revision"])
        for item in snapshot
    }


def _node_dto(node: dict) -> dict:
    return {
        "node_id": node["id"],
        "seq": node["seq"],
        "created_at": node["created_at"],
        "is_baseline": node["is_baseline"],
        "trigger_sortie_id": node["trigger_sortie_id"],
        "trigger_actual_offset": node["trigger_actual_offset"],
        "trigger_revision": node["trigger_revision"],
    }


@app.get("/api/batches/{bid}/report-nodes")
def list_report_nodes(bid: int):
    row = store.get_batch(bid)
    if row is None:
        raise HTTPException(404, "batch not found")
    return {"batch_id": bid, "nodes": [_node_dto(n) for n in store.list_report_nodes(bid)]}


@app.get("/api/batches/{bid}/report-nodes/{node_id}/review")
def review_report_node(bid: int, node_id: int, t: int):
    row = store.get_batch(bid)
    if row is None:
        raise HTTPException(404, "batch not found")
    if t < 0:
        raise HTTPException(400, "t must be a non-negative integer second")
    node = store.get_report_node(bid, node_id)
    if node is None:
        raise HTTPException(404, "report node not found")
    sorties, _ = _build_batch(row)
    before = _reports_from_snapshot(node["before"])
    after = _reports_from_snapshot(node["after"])
    trigger = None
    if not node["is_baseline"]:
        trigger = {
            "sortie_id": node["trigger_sortie_id"],
            "actual_offset": node["trigger_actual_offset"],
            "revision": node["trigger_revision"],
        }
    try:
        result = review_warning(
            sorties, row["plan"]["offsets"], before, after,
            row["safety_radius"], row["safety_height"], t, trigger)
    except EarlyTimeReject as exc:
        raise HTTPException(
            400,
            detail={
                "reason": "t_before_reported_takeoff",
                "t": t,
                "violations": exc.violations,
            },
        )
    return {
        "batch_id": bid,
        "node": {**_node_dto(node),
                 "sortie_id": node["trigger_sortie_id"],
                 "actual_offset": node["trigger_actual_offset"],
                 "revision": node["trigger_revision"]},
        "review": result,
    }


@app.get("/api/health")
def health():
    return {"ok": True}


# Static front-end (served last so /api/* wins).
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/app", StaticFiles(directory=STATIC_DIR, html=True), name="app")


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")

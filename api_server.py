#!/usr/bin/env python3
"""
api_server.py — MktLab Quick Benchmarking Engine
Producción: Railway (backend) + Vercel (frontend)
"""
from __future__ import annotations
import argparse, json, os, pathlib, queue, sys, threading, time, traceback, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from pydantic import BaseModel
except ImportError:
    sys.exit("pip install fastapi uvicorn")

_DIR = Path(__file__).parent
RESULT_FILE = _DIR / "last_result.json"

# ── API key desde variable de entorno (Railway) o desde request (local) ───────
ANTHROPIC_KEY_ENV = os.getenv("ANTHROPIC_API_KEY", "")

# ── Importar pipeline ─────────────────────────────────────────────────────────
def _load_pipeline():
    import importlib.util, sys as s
    for name, fp in [
        ("compare_sites", _DIR / "compare_sites.py"),
        ("analyze_site",  _DIR / "analyze_site.py"),
    ]:
        if name not in s.modules and fp.exists():
            spec = importlib.util.spec_from_file_location(name, fp)
            mod  = importlib.util.module_from_spec(spec)
            s.modules[name] = mod
            spec.loader.exec_module(mod)
    return s.modules.get("compare_sites")

compare_mod = _load_pipeline()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="MktLab Quick Benchmarking Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Job store ─────────────────────────────────────────────────────────────────
class Job:
    def __init__(self, jid: str):
        self.id         = jid
        self.status     = "pending"
        self.events: queue.Queue = queue.Queue()
        self.result     = None
        self.error      = None
        self.created_at = time.time()

JOBS: dict[str, Job] = {}

class CompareRequest(BaseModel):
    urls:               list[str]
    yours_index:        int = 0
    firecrawl_key:      str = ""
    anthropic_key:      str = ""   # si vacío, usa ANTHROPIC_API_KEY del entorno
    extraction_workers: int = 3

# ── SSE ───────────────────────────────────────────────────────────────────────
def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

def _emit(job: Job, event: str, data: Any) -> None:
    job.events.put({"event": event, "data": data})

# ── Pipeline runner ───────────────────────────────────────────────────────────
def _run_pipeline(job: Job, req: CompareRequest) -> None:
    ant_key = req.anthropic_key.strip() or ANTHROPIC_KEY_ENV
    if not ant_key:
        job.status = "error"
        job.error  = "No se encontró ANTHROPIC_API_KEY."
        _emit(job, "error", {"message": job.error})
        job.events.put(None)
        return

    try:
        job.status = "running"
        if not compare_mod:
            raise RuntimeError("compare_sites.py no encontrado junto a api_server.py")

        from compare_sites import (
            BenchmarkState, extract_site,
            run_positioning_agent, run_gaps_agent,
            run_opportunities_agent, run_scoring_agent,
        )

        # Phase 1
        _emit(job, "phase_start", {"phase": 1, "label": "Extrayendo sitios", "total": len(req.urls)})
        state  = BenchmarkState(yours_index=req.yours_index)
        res: dict[int, Any] = {}

        with ThreadPoolExecutor(max_workers=req.extraction_workers) as pool:
            futs = {
                pool.submit(extract_site, url, req.firecrawl_key, ant_key): i
                for i, url in enumerate(req.urls)
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                url = req.urls[idx]
                try:
                    res[idx] = fut.result()
                    chars = res[idx].get("metadata", {}).get("content_length_chars", 0)
                    _emit(job, "site_done", {"index": idx, "url": url, "status": "ok", "chars": chars})
                except Exception as e:
                    res[idx] = e
                    _emit(job, "site_done", {"index": idx, "url": url, "status": "error", "error": str(e)})

        for i in range(len(req.urls)):
            r = res.get(i)
            if isinstance(r, Exception):
                if i == req.yours_index:
                    raise RuntimeError(f"Falló tu sitio ({req.urls[i]}): {r}")
                continue
            state.sites.append(r)

        if len(state.sites) < 2:
            raise RuntimeError("Se necesitan al menos 2 sitios exitosos para comparar.")
        _emit(job, "phase_end", {"phase": 1, "sites_ok": len(state.sites)})

        # Phase 2
        _emit(job, "phase_start", {"phase": 2, "label": "Agentes especialistas", "total": 4})
        agents = {
            "positioning":   run_positioning_agent,
            "gaps":          run_gaps_agent,
            "opportunities": run_opportunities_agent,
            "scoring":       run_scoring_agent,
        }
        ar: dict[str, Any] = {}

        with ThreadPoolExecutor(max_workers=4) as pool:
            futs2 = {pool.submit(fn, state, ant_key): nm for nm, fn in agents.items()}
            for fut in as_completed(futs2):
                nm = futs2[fut]
                try:
                    ar[nm] = fut.result()
                    _emit(job, "agent_done", {"agent": nm, "status": "ok"})
                except Exception as e:
                    ar[nm] = {}
                    _emit(job, "agent_done", {"agent": nm, "status": "error", "error": str(e)})

        state.positioning_report = ar.get("positioning", {})
        state.gaps_report        = ar.get("gaps", {})
        state.opps_report        = ar.get("opportunities", {})
        state.scoring_report     = ar.get("scoring", {})
        _emit(job, "phase_end", {"phase": 2})

        # Phase 3 — ensamblado directo, sin Reporter Agent
        _emit(job, "phase_start", {"phase": 3, "label": "Ensamblando resultado", "total": 1})
        your_url = req.urls[req.yours_index] if req.yours_index < len(req.urls) else ""
        scores   = state.scoring_report.get("scores", []) if state.scoring_report else []
        ys       = next((s for s in scores if s.get("label") == "your_site"), None)

        final = {
            "report_metadata": {
                "generated_at":    datetime.now().isoformat(),
                "sites_analysed":  len(state.sites),
                "your_site_url":   your_url,
                "competitor_urls": [u for i, u in enumerate(req.urls) if i != req.yours_index],
            },
            "executive_summary": (
                f"Benchmark completado para {len(state.sites)} sitios. "
                f"Tu sitio ({your_url}) obtuvo un score de "
                f"{ys.get('total_score', 'N/A') if ys else 'N/A'}/100. "
                "Revisá las secciones de gaps y oportunidades para los próximos pasos."
            ),
            "positioning":   state.positioning_report or {},
            "gaps":          state.gaps_report or {},
            "opportunities": state.opps_report or {},
            "scoring":       state.scoring_report or {},
            "strategic_verdict": {
                "your_site_standing": "challenger",
                "benchmark_leader":   (state.scoring_report or {}).get("benchmark_leader", ""),
                "recommended_focus":  (state.gaps_report or {}).get("gap_summary", "Ver gap analysis"),
            },
            "markdown_report": "",
        }

        try:
            RESULT_FILE.write_text(json.dumps(final, ensure_ascii=False, indent=2))
        except Exception:
            pass

        job.result = final
        job.status = "done"
        _emit(job, "phase_end", {"phase": 3})
        _emit(job, "complete", {
            "job_id":      job.id,
            "total_score": ys.get("total_score") if ys else None,
        })

    except Exception as e:
        job.status = "error"
        job.error  = str(e)
        _emit(job, "error", {"message": str(e), "detail": traceback.format_exc()})
    finally:
        job.events.put(None)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "MktLab Benchmarking Engine"}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    p = _DIR / "compare_ui.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>compare_ui.html no encontrado</h2>", status_code=404)

@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    p = _DIR / "benchmark_dashboard.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>benchmark_dashboard.html no encontrado</h2>", status_code=404)

@app.post("/api/compare")
async def start_compare(req: CompareRequest):
    if len(req.urls) < 2:
        raise HTTPException(400, "Se necesitan al menos 2 URLs.")
    if not req.anthropic_key.strip() and not ANTHROPIC_KEY_ENV:
        raise HTTPException(400, "anthropic_key requerida.")
    jid = str(uuid.uuid4())
    job = Job(jid)
    JOBS[jid] = job
    threading.Thread(target=_run_pipeline, args=(job, req), daemon=True).start()
    return {"job_id": jid, "status": "started"}

@app.get("/api/stream/{job_id}")
async def stream_progress(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado.")
    def gen() -> Generator[str, None, None]:
        yield _sse("status", {"status": job.status})
        while True:
            try:
                msg = job.events.get(timeout=30)
            except queue.Empty:
                yield _sse("heartbeat", {"ts": time.time()})
                continue
            if msg is None:
                break
            yield _sse(msg["event"], msg["data"])
    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado.")
    if job.status == "error":
        raise HTTPException(500, job.error)
    if job.status != "done":
        raise HTTPException(202, "Job en proceso.")
    return JSONResponse(content=job.result)

@app.get("/api/last")
async def get_last():
    done = [(j.created_at, j) for j in JOBS.values() if j.status == "done"]
    if done:
        return JSONResponse(content=sorted(done, key=lambda x: x[0])[-1][1].result)
    if RESULT_FILE.exists():
        return JSONResponse(content=json.loads(RESULT_FILE.read_text()))
    raise HTTPException(404, "No hay resultados todavía.")

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job no encontrado.")
    return {"job_id": job_id, "status": job.status, "error": job.error}

@app.get("/api/key-configured")
async def key_configured():
    """La UI usa esto para saber si la key ya está en el servidor."""
    return {"configured": bool(ANTHROPIC_KEY_ENV)}

@app.delete("/api/jobs")
async def cleanup_jobs():
    cutoff = time.time() - 3600
    old    = [jid for jid, j in JOBS.items() if j.created_at < cutoff]
    for jid in old:
        del JOBS[jid]
    return {"deleted": len(old)}


# ── Entry point local ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        sys.exit("pip install uvicorn")
    pa = argparse.ArgumentParser()
    pa.add_argument("--host", default="127.0.0.1")
    pa.add_argument("--port", type=int, default=8080)
    a = pa.parse_args()
    print(f"\n🦌 MktLab Benchmarking Engine\n   UI   → http://{a.host}:{a.port}\n   Last → http://{a.host}:{a.port}/api/last\n")
    uvicorn.run("api_server:app", host=a.host, port=a.port, log_level="warning")

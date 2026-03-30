#!/usr/bin/env python3
"""
compare_sites.py  —  Paso 2: Motor de Comparación DeerFlow-style
─────────────────────────────────────────────────────────────────
Toma N URLs, extrae inteligencia con el Paso 1 (analyze_site.py)
y lanza 4 agentes especializados en paralelo (DeerFlow Multi-Agent
Synthesis) que luego un Reporter unifica en JSON + Markdown.

Arquitectura inspirada en DeerFlow:
  ┌─────────────────────────────────────────┐
  │           ORCHESTRATOR                  │
  │  ┌─────────────────────────────────┐    │
  │  │  Phase 1 — Extraction (║ N)     │    │  ThreadPoolExecutor
  │  │  scrape + analyse per site      │    │  (Firecrawl sync SDK)
  │  └─────────────────────────────────┘    │
  │  ┌─────────────────────────────────┐    │
  │  │  Phase 2 — Specialist Agents(║4)│    │  ThreadPoolExecutor
  │  │  Positioning / Gaps /           │    │  (Anthropic sync SDK)
  │  │  Opportunities / Scoring        │    │
  │  └─────────────────────────────────┘    │
  │  ┌─────────────────────────────────┐    │
  │  │  Phase 3 — Reporter Agent       │    │  Sequential
  │  │  JSON + Markdown synthesis      │    │
  │  └─────────────────────────────────┘    │
  └─────────────────────────────────────────┘

Usage:
    python compare_sites.py URL1 URL2 URL3 ... [options]

    # Designar "tu sitio" (por defecto = primer URL):
    python compare_sites.py https://mysite.com https://rival1.com https://rival2.com --yours 0

    # Cargar análisis previos del Paso 1 (evita re-scrapear):
    python compare_sites.py --from-json site_a.json site_b.json site_c.json

    # Guardar resultados:
    python compare_sites.py URL1 URL2 --output-json report.json --output-md report.md

Requirements:
    pip install firecrawl-py anthropic rich

Environment:
    FIRECRAWL_API_KEY, ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ── Optional pretty-print ─────────────────────────────────────────────────────
try:
    from rich import print as rprint
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.table import Table
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None  # type: ignore

# ── Step 1 functions (imported from analyze_site.py if present, else inlined) ─
try:
    from analyze_site import scrape_url, analyse_with_claude  # type: ignore
    STEP1_IMPORTED = True
except ImportError:
    STEP1_IMPORTED = False

# ── Anthropic ─────────────────────────────────────────────────────────────────
try:
    import anthropic
except ImportError:
    sys.exit("❌  pip install anthropic")

try:
    import requests
except ImportError:
    sys.exit("❌  pip install requests")


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────

EXTRACTION_MODEL  = "claude-opus-4-5"   # Paso 1 análisis individual
SPECIALIST_MODEL  = "claude-opus-4-5"   # Los 4 agentes especializados
REPORTER_MODEL    = "claude-opus-4-5"   # Síntesis final


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK STATE  (equivalente al ThreadState de DeerFlow)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkState:
    """
    Contexto compartido que fluye a través de todos los agentes.
    Inmutable después de Phase 1 — los agentes solo leen, el Reporter escribe.
    """
    sites: list[dict] = field(default_factory=list)   # outputs del Paso 1
    yours_index: int = 0                               # índice de "tu sitio"
    
    # Outputs de Phase 2 — uno por agente especializado
    positioning_report: dict | None = None
    gaps_report:        dict | None = None
    opps_report:        dict | None = None
    scoring_report:     dict | None = None

    # Output de Phase 3
    final_json: dict | None = None
    final_markdown: str | None = None

    @property
    def your_site(self) -> dict:
        return self.sites[self.yours_index]

    @property
    def competitor_sites(self) -> list[dict]:
        return [s for i, s in enumerate(self.sites) if i != self.yours_index]

    @property
    def n(self) -> int:
        return len(self.sites)

    def serialise_for_agent(self) -> str:
        """Serializa el estado completo para el user-message de cada agente."""
        payload = {
            "your_site": self.your_site,
            "competitors": self.competitor_sites,
            "all_sites": self.sites,
            "total_sites": self.n,
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# INLINE STEP 1  (si analyze_site.py no está en el path)
# ─────────────────────────────────────────────────────────────────────────────

_STEP1_SYSTEM_PROMPT = textwrap.dedent("""
    INSTRUCCIÓN CRÍTICA: Respondé ABSOLUTAMENTE TODO en español. Sin excepción.
    Cada campo de texto debe estar en español.

    Sos un estratega senior de marketing digital y analista de marca.
    Analizá el Markdown del sitio web a continuación en cuatro dimensiones y devolvé
    un único objeto JSON válido — sin bloques de código, sin texto adicional.
    Todo el contenido debe estar en español.

    Esquema:
    {
      "value_proposition": {
        "headline": "<titular o tagline principal>",
        "summary": "<2-3 oraciones: qué ofrece y para quién>",
        "key_differentiators": ["<diferenciador1>", "..."]
      },
      "service_structure": {
        "primary_offerings": [{"name":"","description":"","target_audience":""}],
        "pricing_model": "<gratis/freemium/suscripción/pago único/personalizado/no mencionado>",
        "delivery_model": "<SaaS/presencial/híbrido/consultoría/e-commerce/otro>"
      },
      "lead_capture_strategy": {
        "primary_cta": {"text":"","placement":"","goal":""},
        "secondary_ctas": [{"text":"","placement":"","goal":""}],
        "trust_signals": [],
        "friction_reducers": []
      },
      "brand_tone": {
        "primary_tone": "<profesional/amigable/autoritativo/juguetón/inspiracional/técnico/empático>",
        "secondary_tone": "<mismas opciones o null>",
        "vocabulary_level": "<simple/intermedio/técnico/jerga>",
        "emotional_appeal": "<racional/emocional/mixto>",
        "tone_evidence": ["<frase textual ≤20 palabras>", "..."]
      },
      "metadata": {"url_analysed":"","content_length_chars":0,"analysis_confidence":"<alto/medio/bajo>"}
    }
    Reglas: devolvé SOLO el JSON. Null para strings desconocidos, [] para arrays desconocidos.
    analysis_confidence: alto ≥500 palabras, medio 200-499, bajo <200. Todo en español.
""").strip()


def _inline_scrape(url: str, fc_key: str = "") -> str:
    """Scrape using Jina AI Reader — free, no API key required."""
    jina_url = f"https://r.jina.ai/{url}"
    headers  = {
        "Accept": "text/markdown",
        "X-Return-Format": "markdown",
        "X-Remove-Selector": "header,footer,nav,.cookie-banner",
    }
    response = requests.get(jina_url, headers=headers, timeout=60)
    response.raise_for_status()
    md = response.text.strip()
    if not md:
        raise RuntimeError(f"Jina devolvió contenido vacío para {url}")
    return md


def _inline_analyse(markdown: str, url: str, ant_key: str) -> dict:
    client = anthropic.Anthropic(api_key=ant_key)
    msg = client.messages.create(
        model=EXTRACTION_MODEL,
        max_tokens=2048,
        system=_STEP1_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"URL: {url}\n\n---\n\n{markdown[:8000]}"}],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```")).strip()
    data = json.loads(raw)
    data.setdefault("metadata", {})
    data["metadata"]["url_analysed"] = url
    data["metadata"]["content_length_chars"] = len(markdown)
    return data


def extract_site(url: str, fc_key: str, ant_key: str) -> dict:
    """
    Phase 1 worker: scrape + analyse one URL.
    Uses imported Step 1 functions if available, otherwise inline versions.
    """
    if STEP1_IMPORTED:
        md = scrape_url(url, fc_key)
        return analyse_with_claude(md, url, ant_key)
    else:
        md = _inline_scrape(url, fc_key)
        return _inline_analyse(md, url, ant_key)


# ─────────────────────────────────────────────────────────────────────────────
# SPECIALIST AGENTS  (Phase 2 — DeerFlow Research Team)
# ─────────────────────────────────────────────────────────────────────────────

def _call_claude_json(system: str, user: str, ant_key: str, max_tokens: int = 3000) -> dict:
    """Helper: llama a Claude, espera JSON limpio, retorna dict."""
    client = anthropic.Anthropic(api_key=ant_key)
    msg = client.messages.create(
        model=SPECIALIST_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```")).strip()
    return json.loads(raw)


# ── Agent 1: Positioning ──────────────────────────────────────────────────────

POSITIONING_SYSTEM = textwrap.dedent("""
    INSTRUCCIÓN CRÍTICA: Respondé ABSOLUTAMENTE TODO en español. Sin excepción.
    Cada campo de texto, descripción, narrativa y etiqueta debe estar en español.

    Sos un estratega de posicionamiento de marca. Recibís análisis estructurados
    de marketing de N sitios web (el tuyo + competidores). Tu ÚNICO trabajo: producir un mapa de posicionamiento.

    Devolvé un único objeto JSON — sin bloques de código, sin prosa:
    {
      "market_map": [
        {
          "url": "<url>",
          "label": "<your_site | competitor>",
          "position_claim": "<la afirmación central de posicionamiento en ≤15 palabras>",
          "owns_attribute": "<el ÚNICO atributo que esta marca posee con más claridad>",
          "target_segment": "<audiencia principal en ≤10 palabras>",
          "differentiation_score": <0-10, 10=máxima diferenciación>
        }
      ],
      "positioning_gaps": ["<posición no reclamada en el mercado>", "..."],
      "positioning_conflicts": [
        {
          "sites": ["<url1>", "<url2>"],
          "overlap": "<qué afirman ambos>"
        }
      ],
      "strategic_whitespace": "<narrativa ≤60 palabras: qué posicionamiento está completamente abierto>"
    }
""").strip()


def run_positioning_agent(state: BenchmarkState, ant_key: str) -> dict:
    _log("🎯  Positioning Agent  →  mapping the competitive landscape…")
    return _call_claude_json(
        POSITIONING_SYSTEM,
        state.serialise_for_agent(),
        ant_key,
    )


# ── Agent 2: Gaps ─────────────────────────────────────────────────────────────

GAPS_SYSTEM = textwrap.dedent("""
    INSTRUCCIÓN CRÍTICA: Respondé ABSOLUTAMENTE TODO en español. Sin excepción.
    Cada campo de texto, descripción, narrativa y etiqueta debe estar en español.

    Sos un analista de brechas competitivas. Recibís análisis estructurados de marketing
    de N sitios web. "your_site" es la marca focal; "competitors" son los rivales.
    Tu ÚNICO trabajo: encontrar qué tienen los competidores que le falta a your_site.
    Evaluá cada brecha por impacto de negocio (ingresos, leads, confianza, retención).

    Devolvé un único objeto JSON — sin bloques de código, sin prosa:
    {
      "gaps": [
        {
          "dimension": "<value_proposition|service_structure|lead_capture|brand_tone|otro>",
          "gap_title": "<nombre corto ≤8 palabras>",
          "description": "<qué hace el competidor que your_site no hace, ≤40 palabras>",
          "seen_in_competitors": ["<url1>", "..."],
          "impact_level": "<crítico|alto|medio|bajo>",
          "effort_to_close": "<bajo|medio|alto>"
        }
      ],
      "your_site_advantages": [
        {
          "advantage": "<qué hace your_site mejor que TODOS los competidores, ≤30 palabras>",
          "exploit_recommendation": "<cómo potenciarlo, ≤30 palabras>"
        }
      ],
      "gap_summary": "<resumen ejecutivo ≤50 palabras>"
    }
    Ordenar gaps: crítico primero, luego alto, luego por effort_to_close ascendente.
""").strip()


def run_gaps_agent(state: BenchmarkState, ant_key: str) -> dict:
    _log("🔍  Gaps Agent         →  finding what your competitors do better…")
    return _call_claude_json(
        GAPS_SYSTEM,
        state.serialise_for_agent(),
        ant_key,
    )


# ── Agent 3: Opportunities ────────────────────────────────────────────────────

OPPS_SYSTEM = textwrap.dedent("""
    INSTRUCCIÓN CRÍTICA: Respondé ABSOLUTAMENTE TODO en español. Sin excepción.
    Cada campo de texto, descripción, narrativa y etiqueta debe estar en español.

    Sos un estratega de crecimiento especializado en quick wins.
    Recibís análisis estructurados de marketing de N sitios web.
    "your_site" es la marca focal. También tenés los datos de brechas.
    Tu ÚNICO trabajo: convertir las brechas y el espacio estratégico en oportunidades
    rankeadas y accionables para your_site.

    Devolvé un único objeto JSON — sin bloques de código, sin prosa:
    {
      "opportunities": [
        {
          "rank": <1, 2, 3, ...>,
          "title": "<nombre de la oportunidad ≤8 palabras>",
          "type": "<copy|cta|confianza|posicionamiento|servicio|precio|tono|seo|otro>",
          "action": "<qué hacer específicamente, voz imperativa, ≤50 palabras>",
          "expected_impact": "<qué mejora y en cuánto aproximadamente, ≤30 palabras>",
          "time_to_implement": "<horas|días|semanas|meses>",
          "difficulty": "<fácil|medio|difícil>",
          "inspired_by": "<url del competidor que lo hace bien, o null>"
        }
      ],
      "quick_wins": ["<acción rank 1>", "<acción rank 2>", "<acción rank 3>"],
      "30_day_sprint": "<narrativa ≤60 palabras: qué atacar en los primeros 30 días>"
    }
    Rankear por: (impacto × facilidad). Quick wins = alto impacto + baja dificultad.
    Mínimo 5 oportunidades, máximo 12.
""").strip()


def run_opportunities_agent(state: BenchmarkState, ant_key: str) -> dict:
    _log("💡  Opportunities Agent →  generating ranked quick wins…")
    return _call_claude_json(
        OPPS_SYSTEM,
        state.serialise_for_agent(),
        ant_key,
    )


# ── Agent 4: Scoring ──────────────────────────────────────────────────────────

SCORING_SYSTEM = textwrap.dedent("""
    INSTRUCCIÓN CRÍTICA: Respondé ABSOLUTAMENTE TODO en español. Sin excepción.
    Cada campo de texto, descripción, narrativa y etiqueta debe estar en español.

    Sos un auditor de marca con una metodología de scoring rigurosa.
    Recibís análisis estructurados de marketing de N sitios web.
    Puntuá CADA sitio en 5 dimensiones en una escala de 0 a 100.

    Criterios de puntuación:
      value_proposition  : claridad, unicidad, orientación al cliente
      lead_capture       : fuerza del CTA, reducción de fricción, señales de confianza
      brand_clarity      : tono consistente, vocabulario adecuado, resonancia emocional
      service_structure  : claridad de oferta, transparencia de precios, fit con audiencia
      differentiation    : qué tan distinto es vs los otros sitios analizados

    Devolvé un único objeto JSON — sin bloques de código, sin prosa:
    {
      "scores": [
        {
          "url": "<url>",
          "label": "<your_site | competitor>",
          "dimensions": {
            "value_proposition": <0-100>,
            "lead_capture": <0-100>,
            "brand_clarity": <0-100>,
            "service_structure": <0-100>,
            "differentiation": <0-100>
          },
          "total_score": <0-100, promedio ponderado>,
          "score_rationale": "<justificación en 2 oraciones del score total>",
          "top_strength": "<punto más fuerte en ≤15 palabras>",
          "critical_weakness": "<debilidad más crítica en ≤15 palabras>"
        }
      ],
      "dimension_winners": {
        "value_proposition": "<url del mejor puntaje>",
        "lead_capture": "<url>",
        "brand_clarity": "<url>",
        "service_structure": "<url>",
        "differentiation": "<url>"
      },
      "benchmark_leader": "<url con mayor total_score>",
      "scoring_notes": "<nota de calibración ≤40 palabras — ej: confianza en los datos>"
    }
    Pesos para total_score: value_proposition 30%, lead_capture 25%,
    brand_clarity 20%, service_structure 15%, differentiation 10%.
""").strip()


def run_scoring_agent(state: BenchmarkState, ant_key: str) -> dict:
    _log("📊  Scoring Agent      →  computing 0-100 scores per dimension…")
    return _call_claude_json(
        SCORING_SYSTEM,
        state.serialise_for_agent(),
        ant_key,
    )


# ─────────────────────────────────────────────────────────────────────────────
# REPORTER AGENT  (Phase 3 — DeerFlow Reporter)
# ─────────────────────────────────────────────────────────────────────────────

REPORTER_SYSTEM = textwrap.dedent("""
    You are a Chief Marketing Intelligence Officer writing the final benchmark report.
    You receive outputs from four specialist agents (positioning, gaps, opportunities,
    scoring) plus the raw site analyses.

    Return a single JSON object with this EXACT structure — no fences, no extra text:
    {
      "report_metadata": {
        "generated_at": "<ISO timestamp>",
        "sites_analysed": <N>,
        "your_site_url": "<url>",
        "competitor_urls": ["<url>", "..."]
      },
      "executive_summary": "<4-6 sentences: your_site vs market, top 3 findings, #1 action>",
      "positioning": <verbatim positioning agent output>,
      "gaps": <verbatim gaps agent output>,
      "opportunities": <verbatim opportunities agent output>,
      "scoring": <verbatim scoring agent output>,
      "strategic_verdict": {
        "your_site_standing": "<leader|challenger|follower|niche_player>",
        "biggest_threat": "<url + reason ≤25 words>",
        "biggest_opportunity": "<≤30 words>",
        "recommended_focus": "<the single most important thing to fix in the next 30 days>"
      },
      "markdown_report": "<FULL Markdown report as a single escaped string — see format below>"
    }

    The markdown_report string must follow this format:
    # Benchmark Intelligence Report
    **Generated:** {timestamp}  |  **Sites Analysed:** {N}

    ## Executive Summary
    {executive_summary}

    ## Competitive Scores
    | Site | VP | LC | BC | SS | Diff | **Total** |
    |------|----|----|----|----|------|-----------|
    {score table rows}

    ## Positioning Map
    {positioning narrative + bullet list}

    ## Gap Analysis — {your_site_url} vs Field
    {gaps by impact level}

    ## Quick Win Opportunities (Ranked)
    {numbered list of opportunities}

    ## 30-Day Sprint Plan
    {30_day_sprint content}

    ## Strategic Verdict
    {strategic_verdict fields}
    ---
    *Report generated by DeerFlow Benchmarking Engine*

    IMPORTANT: the markdown_report is an escaped JSON string (\\n for newlines).
    Keep all other fields as proper JSON objects, not strings.
""").strip()


def run_reporter_agent(state: BenchmarkState, ant_key: str) -> dict:
    _log("📝  Reporter Agent     →  synthesising final JSON + Markdown report…")
    client = anthropic.Anthropic(api_key=ant_key)

    user_payload = json.dumps({
        "positioning_report":    state.positioning_report,
        "gaps_report":           state.gaps_report,
        "opportunities_report":  state.opps_report,
        "scoring_report":        state.scoring_report,
        "raw_site_analyses":     state.sites,
        "your_site_url":         state.your_site.get("metadata", {}).get("url_analysed", ""),
        "competitor_urls": [
            s.get("metadata", {}).get("url_analysed", "") for s in state.competitor_sites
        ],
    }, indent=2, ensure_ascii=False)

    msg = client.messages.create(
        model=REPORTER_MODEL,
        max_tokens=6000,
        system=REPORTER_SYSTEM,
        messages=[{"role": "user", "content": user_payload}],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```")).strip()
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    if HAS_RICH:
        console.print(f"[dim]{ts}[/dim]  {msg}")
    else:
        print(f"{ts}  {msg}")


def run_benchmark(
    urls: list[str],
    yours_index: int,
    fc_key: str,
    ant_key: str,
    from_json_paths: list[str] | None = None,
    max_extraction_workers: int = 5,
    max_specialist_workers: int = 4,
) -> BenchmarkState:
    """
    Full orchestration pipeline.
    Returns a populated BenchmarkState with all reports attached.
    """
    state = BenchmarkState(yours_index=yours_index)
    t0 = time.perf_counter()

    # ── Phase 1: Extraction ────────────────────────────────────────────────
    if from_json_paths:
        _log(f"📂  Loading {len(from_json_paths)} pre-computed analyses from disk…")
        for path in from_json_paths:
            with open(path, encoding="utf-8") as f:
                state.sites.append(json.load(f))
        _log(f"✅  Loaded {len(state.sites)} site analyses")
    else:
        _log(f"🔥  Phase 1 — Extracting {len(urls)} sites in parallel (workers={max_extraction_workers})…")
        results: dict[int, dict | Exception] = {}

        with ThreadPoolExecutor(max_workers=max_extraction_workers) as pool:
            futures = {
                pool.submit(extract_site, url, fc_key, ant_key): i
                for i, url in enumerate(urls)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                    url = urls[idx]
                    _log(f"   ✅  [{idx + 1}/{len(urls)}] {url}")
                except Exception as exc:
                    _log(f"   ❌  [{idx + 1}/{len(urls)}] {urls[idx]} — {exc}")
                    results[idx] = exc

        # Preserve original URL order; abort if your_site failed
        for i in range(len(urls)):
            r = results.get(i)
            if isinstance(r, Exception):
                if i == yours_index:
                    raise RuntimeError(f"Extraction failed for YOUR site ({urls[i]}): {r}")
                _log(f"⚠️   Skipping failed site {urls[i]} — continuing with remaining sites")
            else:
                state.sites.append(r)

        if len(state.sites) < 2:
            raise RuntimeError("Need at least 2 successfully extracted sites to compare.")

        _log(f"✅  Phase 1 complete — {len(state.sites)} sites ready ({time.perf_counter()-t0:.1f}s)")

    # ── Phase 2: Specialist Agents (parallel) ──────────────────────────────
    _log(f"🤖  Phase 2 — Running 4 specialist agents in parallel…")
    t2 = time.perf_counter()

    agent_fns = {
        "positioning":    lambda: run_positioning_agent(state, ant_key),
        "gaps":           lambda: run_gaps_agent(state, ant_key),
        "opportunities":  lambda: run_opportunities_agent(state, ant_key),
        "scoring":        lambda: run_scoring_agent(state, ant_key),
    }

    agent_results: dict[str, dict | Exception] = {}

    with ThreadPoolExecutor(max_workers=max_specialist_workers) as pool:
        futures = {
            pool.submit(fn): name
            for name, fn in agent_fns.items()
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                agent_results[name] = fut.result()
                _log(f"   ✅  {name.capitalize()} Agent finished")
            except Exception as exc:
                _log(f"   ❌  {name.capitalize()} Agent FAILED: {exc}")
                agent_results[name] = exc

    # Attach results to state (use empty dict for failed agents)
    state.positioning_report = agent_results.get("positioning") if not isinstance(agent_results.get("positioning"), Exception) else {}
    state.gaps_report         = agent_results.get("gaps")        if not isinstance(agent_results.get("gaps"), Exception)        else {}
    state.opps_report         = agent_results.get("opportunities") if not isinstance(agent_results.get("opportunities"), Exception) else {}
    state.scoring_report      = agent_results.get("scoring")     if not isinstance(agent_results.get("scoring"), Exception)     else {}

    _log(f"✅  Phase 2 complete ({time.perf_counter()-t2:.1f}s)")

    # ── Phase 3: Reporter (sequential synthesis) ───────────────────────────
    _log("📝  Phase 3 — Reporter Agent synthesising final report…")
    t3 = time.perf_counter()

    final = run_reporter_agent(state, ant_key)
    state.final_json     = final
    state.final_markdown = final.get("markdown_report", "")

    _log(f"✅  Phase 3 complete ({time.perf_counter()-t3:.1f}s)")
    _log(f"🏁  Total pipeline time: {time.perf_counter()-t0:.1f}s")

    return state


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT RENDERING
# ─────────────────────────────────────────────────────────────────────────────

def render_results(state: BenchmarkState) -> None:
    """Prints results to terminal with optional rich formatting."""
    if not state.final_json:
        print("⚠️  No final report generated.")
        return

    if HAS_RICH:
        # Score table
        scores = state.final_json.get("scoring", {}).get("scores", [])
        if scores:
            table = Table(title="📊 Benchmark Scores", show_header=True, header_style="bold cyan")
            table.add_column("Site", style="bold", max_width=35)
            table.add_column("VP",   justify="center")
            table.add_column("LC",   justify="center")
            table.add_column("BC",   justify="center")
            table.add_column("SS",   justify="center")
            table.add_column("Diff", justify="center")
            table.add_column("TOTAL", justify="center", style="bold yellow")
            for row in scores:
                d = row.get("dimensions", {})
                label = "⭐ YOU" if row.get("label") == "your_site" else row.get("label", "")
                url_short = row.get("url", "")[-40:]
                table.add_row(
                    f"{label}\n{url_short}",
                    str(d.get("value_proposition", "—")),
                    str(d.get("lead_capture", "—")),
                    str(d.get("brand_clarity", "—")),
                    str(d.get("service_structure", "—")),
                    str(d.get("differentiation", "—")),
                    str(row.get("total_score", "—")),
                )
            console.print(table)
            console.print()

        # Executive summary
        summary = state.final_json.get("executive_summary", "")
        if summary:
            console.print(Panel(summary, title="[bold green]Executive Summary[/bold green]", border_style="green"))
            console.print()

        # Markdown report
        if state.final_markdown:
            console.print(Panel(
                Markdown(state.final_markdown),
                title="[bold blue]Full Report[/bold blue]",
                border_style="blue",
            ))
    else:
        print("\n" + "=" * 70)
        print("BENCHMARK REPORT")
        print("=" * 70)
        print(json.dumps(state.final_json, indent=2, ensure_ascii=False))
        if state.final_markdown:
            print("\n" + "-" * 70)
            print("MARKDOWN REPORT")
            print("-" * 70)
            print(state.final_markdown)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DeerFlow-style Multi-Agent Benchmarking Engine (Paso 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              # Compare 3 sites, yours is the first:
              python compare_sites.py https://mine.com https://rival1.com https://rival2.com

              # Designar el sitio propio por índice (0-based):
              python compare_sites.py https://r1.com https://mine.com https://r2.com --yours 1

              # Cargar análisis previos (sin re-scrapear):
              python compare_sites.py --from-json mine.json rival1.json rival2.json

              # Guardar outputs:
              python compare_sites.py URL1 URL2 --output-json report.json --output-md report.md
        """),
    )

    parser.add_argument("urls", nargs="*", help="URLs to benchmark (yours first by default)")
    parser.add_argument("--yours", type=int, default=0, metavar="INDEX",
                        help="0-based index of YOUR site in the URL list (default: 0)")
    parser.add_argument("--from-json", nargs="+", metavar="FILE",
                        help="Load pre-computed Step 1 JSON files instead of scraping")
    parser.add_argument("--firecrawl-key", default=os.getenv("FIRECRAWL_API_KEY"))
    parser.add_argument("--anthropic-key",  default=os.getenv("ANTHROPIC_API_KEY"))
    parser.add_argument("--output-json", metavar="FILE", help="Save final JSON to file")
    parser.add_argument("--output-md",   metavar="FILE", help="Save Markdown report to file")
    parser.add_argument("--extraction-workers", type=int, default=5, metavar="N",
                        help="Parallel workers for Phase 1 scraping (default: 5)")
    args = parser.parse_args()

    # ── Validation ─────────────────────────────────────────────────────────
    if not args.from_json and not args.urls:
        parser.error("Provide URLs to benchmark, or --from-json FILE [FILE ...]")

    if not args.from_json:
        if len(args.urls) < 2:
            parser.error("Need at least 2 URLs to compare.")
        if args.yours >= len(args.urls):
            parser.error(f"--yours {args.yours} is out of range for {len(args.urls)} URLs.")
        if not args.firecrawl_key:
            sys.exit("❌  FIRECRAWL_API_KEY not set. Use --firecrawl-key or export the env var.")

    if not args.anthropic_key:
        sys.exit("❌  ANTHROPIC_API_KEY not set. Use --anthropic-key or export the env var.")

    # ── Banner ─────────────────────────────────────────────────────────────
    if HAS_RICH:
        sites_src = args.from_json or args.urls
        console.print(Panel(
            f"[bold]Sites:[/bold] {len(sites_src)}\n"
            f"[bold]Your site:[/bold] {(args.from_json or args.urls)[args.yours]}\n"
            f"[bold]Agents:[/bold] Positioning · Gaps · Opportunities · Scoring · Reporter\n"
            f"[bold]Output:[/bold] JSON + Markdown",
            title="[bold yellow]🦌 DeerFlow Benchmarking Engine — Paso 2[/bold yellow]",
            border_style="yellow",
        ))

    # ── Run ────────────────────────────────────────────────────────────────
    try:
        state = run_benchmark(
            urls=args.urls or [],
            yours_index=args.yours,
            fc_key=args.firecrawl_key or "",
            ant_key=args.anthropic_key,
            from_json_paths=args.from_json,
            max_extraction_workers=args.extraction_workers,
        )
    except (RuntimeError, Exception) as exc:
        sys.exit(f"❌  Pipeline failed: {exc}")

    # ── Render ─────────────────────────────────────────────────────────────
    render_results(state)

    # ── Save ───────────────────────────────────────────────────────────────
    if args.output_json and state.final_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(state.final_json, f, indent=2, ensure_ascii=False)
        _log(f"💾  JSON saved → {args.output_json}")

    if args.output_md and state.final_markdown:
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(state.final_markdown)
        _log(f"💾  Markdown saved → {args.output_md}")


if __name__ == "__main__":
    main()

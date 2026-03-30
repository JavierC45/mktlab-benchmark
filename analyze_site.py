#!/usr/bin/env python3
"""
analyze_site.py
---------------
Scrapes a URL using Jina AI Reader (free, no API key needed) and sends the
resulting Markdown to Claude (Anthropic) for a structured marketing analysis.

Usage:
    python analyze_site.py <URL> [--anthropic-key KEY]

Environment variables (alternative to flags):
    ANTHROPIC_API_KEY

Requirements:
    pip install anthropic requests
"""

import argparse
import json
import os
import sys
import textwrap
import requests

# ── Optional: rich pretty-printing ────────────────────────────────────────────
try:
    from rich import print as rprint
    from rich.panel import Panel
    from rich.syntax import Syntax
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# ── Third-party ───────────────────────────────────────────────────────────────
try:
    import anthropic
except ImportError:
    sys.exit(
        "❌  Missing dependency: run  pip install anthropic requests  and try again."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CLAUDE_MODEL = "claude-opus-4-5"   # change to "claude-3-5-haiku-20241022" for faster/cheaper runs

SYSTEM_PROMPT = textwrap.dedent("""
    INSTRUCCIÓN CRÍTICA: Respondé ABSOLUTAMENTE TODO en español. Sin excepción.
    Cada campo de texto, descripción, resumen, evidencia y etiqueta debe estar en español.

    Sos un estratega senior de marketing digital y analista de marca.
    Vas a recibir el contenido Markdown de la landing page de un sitio web.
    Tu tarea es analizarlo en exactamente cuatro dimensiones y devolver un
    único objeto JSON válido — sin bloques de código, sin texto adicional.
    Responde TODO en español, incluyendo descripciones, resúmenes y evidencias.

    El JSON debe seguir este esquema exacto:

    {
      "value_proposition": {
        "headline": "<titular o tagline principal detectado>",
        "summary": "<descripción en 2-3 oraciones de qué ofrece la empresa y a quién>",
        "key_differentiators": ["<diferenciador 1>", "<diferenciador 2>", "..."]
      },
      "service_structure": {
        "primary_offerings": [
          {
            "name": "<nombre del servicio/producto>",
            "description": "<descripción breve>",
            "target_audience": "<a quién está dirigido>"
          }
        ],
        "pricing_model": "<gratis/freemium/suscripción/pago único/personalizado/no mencionado>",
        "delivery_model": "<SaaS/presencial/híbrido/consultoría/e-commerce/otro>"
      },
      "lead_capture_strategy": {
        "primary_cta": {
          "text": "<texto exacto del CTA>",
          "placement": "<hero/navbar/footer/sticky/popup/otro>",
          "goal": "<prueba/demo/contacto/compra/newsletter/otro>"
        },
        "secondary_ctas": [
          {
            "text": "<texto del CTA>",
            "placement": "<ubicación>",
            "goal": "<objetivo>"
          }
        ],
        "trust_signals": ["<ej: testimonios, logos, certificaciones, casos de éxito>"],
        "friction_reducers": ["<ej: sin tarjeta de crédito, prueba gratis, garantía de devolución>"]
      },
      "brand_tone": {
        "primary_tone": "<uno de: profesional / amigable / autoritativo / juguetón / inspiracional / técnico / empático>",
        "secondary_tone": "<una de las mismas opciones, o null>",
        "vocabulary_level": "<simple / intermedio / técnico / cargado de jerga>",
        "emotional_appeal": "<racional / emocional / mixto>",
        "tone_evidence": ["<frase textual de la página que ilustra el tono, máx 20 palabras>", "..."]
      },
      "metadata": {
        "url_analysed": "<la URL analizada>",
        "content_length_chars": 0,
        "analysis_confidence": "<alto / medio / bajo>"
      }
    }

    Reglas:
    - Devolvé SOLO el objeto JSON. Sin prosa, sin bloques de código markdown.
    - Si un campo no puede determinarse, usá null para strings o [] para arrays.
    - tone_evidence debe contener al menos 2 frases textuales reales del Markdown, cada una de menos de 20 palabras.
    - analysis_confidence es "alto" si la página tiene ≥ 500 palabras, "medio" para 200-499, "bajo" para < 200.
    - Todo el contenido de texto debe estar en español.
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2. FIRECRAWL SCRAPER
# ─────────────────────────────────────────────────────────────────────────────

def scrape_url(url: str, api_key: str = "") -> str:
    """
    Scrapes *url* using Jina AI Reader (free, no API key needed).
    Raises RuntimeError on failure.
    """
    print(f"🌐  Jina → scraping {url} …")

    jina_url = f"https://r.jina.ai/{url}"
    headers  = {
        "Accept": "text/markdown",
        "X-Return-Format": "markdown",
        "X-Remove-Selector": "header,footer,nav,.cookie-banner",
    }

    try:
        response = requests.get(jina_url, headers=headers, timeout=60)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Jina tardó más de 60s en responder para {url}")
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Error al contactar Jina: {exc}")

    markdown = response.text.strip()

    if not markdown:
        raise RuntimeError(
            f"Jina devolvió contenido vacío para {url}. "
            "La página puede estar bloqueada o requerir JavaScript."
        )

    print(f"✅  Markdown recibido — {len(markdown):,} caracteres")
    return markdown


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLAUDE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_with_claude(markdown: str, url: str, api_key: str) -> dict:
    """
    Sends the Markdown to Claude and returns the parsed JSON analysis dict.
    """
    print(f"🤖  Claude ({CLAUDE_MODEL}) → analysing content …")

    client = anthropic.Anthropic(api_key=api_key)

    user_message = (
        f"URL: {url}\n\n"
        "Below is the full Markdown scraped from the page. "
        "Analyse it and return the JSON object as instructed.\n\n"
        "---\n\n"
        f"{markdown[:8000]}"
    )

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = message.content[0].text.strip()

    # Strip accidental markdown code fences if the model adds them
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        raw_text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        analysis = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Claude response is not valid JSON.\n"
            f"Parse error: {exc}\n"
            f"Raw response (first 500 chars):\n{raw_text[:500]}"
        ) from exc

    # Inject real metadata values
    analysis.setdefault("metadata", {})
    analysis["metadata"]["url_analysed"] = url
    analysis["metadata"]["content_length_chars"] = len(markdown)

    print("✅  Analysis complete")
    return analysis


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape a URL with Jina AI and analyse it with Claude."
    )
    parser.add_argument("url", help="Target URL to analyse")
    parser.add_argument(
        "--anthropic-key",
        default=os.getenv("ANTHROPIC_API_KEY"),
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the JSON result (e.g. result.json)",
    )
    args = parser.parse_args()

    if not args.anthropic_key:
        sys.exit(
            "❌  No Anthropic API key found. "
            "Pass --anthropic-key or set ANTHROPIC_API_KEY."
        )

    # ── Run pipeline ───────────────────────────────────────────────────────
    try:
        markdown = scrape_url(args.url)
        analysis = analyse_with_claude(markdown, args.url, args.anthropic_key)
    except RuntimeError as exc:
        sys.exit(f"❌  {exc}")

    # ── Output ─────────────────────────────────────────────────────────────
    json_str = json.dumps(analysis, indent=2, ensure_ascii=False)

    if HAS_RICH:
        rprint(Panel(
            Syntax(json_str, "json", theme="monokai", line_numbers=False),
            title="[bold green]Site Analysis[/bold green]",
            border_style="green",
        ))
    else:
        print("\n" + "=" * 60)
        print("SITE ANALYSIS RESULT")
        print("=" * 60)
        print(json_str)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(json_str)
        print(f"\n💾  Result saved to: {args.output}")

    # Return the dict programmatically when imported as a module
    return analysis


if __name__ == "__main__":
    main()

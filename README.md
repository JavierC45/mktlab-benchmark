# 🦌 MktLab Quick Benchmarking Engine

Herramienta de benchmarking competitivo con IA. Analiza N sitios web en paralelo usando Jina AI (scraping) + Claude (análisis), con 4 agentes especializados y un dashboard visual.

## Arquitectura

```
GitHub
├── Backend (Python/FastAPI)  →  Railway
└── Frontend (HTML)           →  servido por el mismo backend
```

## Deploy en Railway

### 1. Crear cuenta
→ https://railway.app (plan Hobby: $5/mes, sin timeout de requests)

### 2. Nuevo proyecto desde GitHub
- New Project → Deploy from GitHub repo
- Seleccionar este repositorio
- Railway detecta automáticamente el `Procfile`

### 3. Configurar variable de entorno
En Railway → tu proyecto → Variables:
```
ANTHROPIC_API_KEY = sk-ant-xxxxxxxxxx
```

### 4. Deploy automático
Railway hace deploy en cada `git push` a `main`.

### 5. Acceder a la app
Railway te da una URL pública tipo:
```
https://mktlab-benchmark-production.up.railway.app
```

## Estructura del repo

```
├── api_server.py          # Backend FastAPI (Railway)
├── compare_sites.py       # Pipeline multi-agente
├── analyze_site.py        # Extractor individual (Jina + Claude)
├── compare_ui.html        # UI principal
├── benchmark_dashboard.html # Dashboard de resultados
├── requirements.txt       # Dependencias Python
├── Procfile               # Comando de inicio para Railway
├── railway.toml           # Configuración Railway
└── .gitignore
```

## Variables de entorno

| Variable | Descripción | Requerida |
|---|---|---|
| `ANTHROPIC_API_KEY` | API key de Anthropic (Claude) | ✅ Sí |
| `PORT` | Puerto (Railway lo inyecta automáticamente) | Auto |

## Desarrollo local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Correr servidor
python api_server.py

# Abrir en browser
open http://localhost:8080
```

## Tecnologías

- **Scraping**: Jina AI Reader (gratis, sin API key)
- **Análisis**: Claude Haiku (Anthropic)
- **Backend**: FastAPI + uvicorn
- **Deploy**: Railway
- **Frontend**: HTML/CSS/JS vanilla

---
© 2026 R'Evolution Education Group TM · Powered by MktLab

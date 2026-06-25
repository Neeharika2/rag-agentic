# Placement Intelligence Assistant

Multi-agent RAG system for engineering placement queries. Built with LangGraph, LangChain, ChromaDB, and FastAPI.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
python run_ingestion.py  # index the PDF dataset (one-time)
python run.py            # start web UI at http://127.0.0.1:8000
python run.py --cli "CGPA cutoff for TCS"  # CLI mode
```

## Features

- **10 specialized agent nodes** — eligibility, interview prep, hiring stats, trends, conflict resolution, strategy, simulation, web search
- **Digital Twin "What-If" Simulation** — modify CGPA/skills and see projected readiness changes
- **3-column dashboard** — profile editor, chat console, readiness analytics with trend charts
- **Multi-hop reasoning** — joins data across eligibility, hiring, stats, and trend sections
- **Conflict detection** — flags discrepancies between official and portal data sources
- **LLM-as-a-judge evaluation** — 30-query test suite with routing and factual accuracy checks

## Architecture

```
profile_builder → router → [eligibility|interview|hiring|stats|trend|websearch]
                         → [opportunity → probability_estimator]  (strategy/simulation)
                                ↓
                         validation → synthesis → END
```

Detailed docs in [`project_phases/README.md`](project_phases/README.md).

## Environment

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Google Gemini API key (fallback LLM) |
| `DEEPSEEK_API_KEY` | Yes | DeepSeek API key (primary LLM) |
| `TAVILY_API_KEY` | Yes | Tavily web search API key |
| `DEEPSEEK_MODEL` | No | Default: `deepseek-v4-flash` |

## Project Structure

```
├── agent_nodes/        # LangGraph agent nodes
├── evaluation/         # 30-query test suite
├── parsing/            # PDF parser (Docling)
├── vectorstore/        # ChromaDB wrapper
├── static/             # Frontend (vanilla HTML/CSS/JS)
├── uploads/            # PDF dataset
├── run.py              # CLI/Web runner
├── run_ingestion.py    # Data ingestion
└── rag_pipeline.py     # Graph definition
```

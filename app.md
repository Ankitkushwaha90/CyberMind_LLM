### === File: README.md ===
# CyberMind: Intelligent Threat Analyzer Using LLMs

## Features:
- Collect system info
- Analyze CVE relevance
- Use LLM to explain and suggest security actions

## Getting Started:
```bash
pip install fastapi uvicorn transformers langchain faiss-cpu psutil requests
uvicorn backend.main:app --reload
python ui/cli.py
```

## Customize:
- Replace LLM in `model_interface.py`
- Add more CVEs to `data/cve_data.json`
- Fine-tune your own LLM in `ml_model/`

## Goal:
Learn how LLMs + vector search + system info can combine to create intelligent cybersecurity tools.

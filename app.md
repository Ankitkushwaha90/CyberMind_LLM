### === File: backend/main.py ===
from fastapi import FastAPI, Request
from backend.analyzer import analyze_system

app = FastAPI()

@app.get("/analyze")
async def analyze():
    result = analyze_system()
    return result


### === File: backend/analyzer.py ===
import psutil, socket, platform
from backend.model_interface import query_llm
from backend.data_loader import load_cve_data, search_vulnerabilities

def gather_info():
    return {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "os_version": platform.version(),
        "cpu_usage": psutil.cpu_percent(),
        "memory": psutil.virtual_memory()._asdict(),
    }

def analyze_system():
    info = gather_info()
    cve_matches = search_vulnerabilities(info)
    explanation = query_llm(f"Analyze these vulnerabilities: {cve_matches}")
    return {
        "system_info": info,
        "vulnerabilities": cve_matches,
        "llm_analysis": explanation
    }


### === File: backend/model_interface.py ===
from transformers import pipeline

# Use local model or Hugging Face hosted
llm = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1")

def query_llm(prompt):
    output = llm(prompt, max_new_tokens=256)[0]['generated_text']
    return output


### === File: backend/data_loader.py ===
import json
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

cve_data_path = "data/cve_data.json"
faiss_index = "data/faiss_index"

embeddings = HuggingFaceEmbeddings()
db = FAISS.load_local(faiss_index, embeddings)

def load_cve_data():
    with open(cve_data_path, "r") as f:
        return json.load(f)

def search_vulnerabilities(info):
    query = f"OS: {info['platform']} {info['os_version']}, CPU: {info['cpu_usage']}%"
    return db.similarity_search(query)


### === File: ui/cli.py ===
import requests

def main():
    print("[+] Analyzing system...")
    res = requests.get("http://127.0.0.1:8000/analyze")
    data = res.json()
    print("System Info:", data["system_info"])
    print("Vulnerabilities:", data["vulnerabilities"])
    print("LLM Analysis:\n", data["llm_analysis"])

if __name__ == '__main__':
    main()


### === File: ml_model/prompts.py ===
def generate_prompt(vuln_data):
    return f"""You are a cybersecurity expert.
Analyze these system vulnerabilities:
{vuln_data}
Explain the threats and recommend mitigation."""


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

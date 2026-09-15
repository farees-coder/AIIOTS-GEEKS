@echo off
:: ═══════════════════════════════════════════════════════════
::  PDF Q&A — RAG + Groq  |  Launch Script
:: ═══════════════════════════════════════════════════════════

:: Suppress HuggingFace online checks after first model download
set TRANSFORMERS_OFFLINE=0
set HF_DATASETS_OFFLINE=0

:: Suppress tokenizer parallelism warning (prevents deadlocks)
set TOKENIZERS_PARALLELISM=false

:: Suppress transformers warnings about missing torchvision
set TRANSFORMERS_NO_ADVISORY_WARNINGS=1

:: Reduce OpenBLAS thread count to ease memory pressure
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1

echo.
echo  =========================================
echo   PDF Q^&A with RAG + Groq
echo   Starting at http://localhost:8501
echo  =========================================
echo.

call venv\Scripts\streamlit.exe run app.py ^
    --server.port 8501 ^
    --server.headless false ^
    --server.fileWatcherType none ^
    --browser.gatherUsageStats false

pause

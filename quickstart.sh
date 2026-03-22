#!/bin/bash
# ArvyaX Quick Start - Run this to test the system

set -e

PROJECT_DIR="/home/rajeev/Projects/Arvyax_Assignment"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"

echo "🌿 ArvyaX Emotional Intelligence System"
echo "======================================"
echo ""

# Activate venv
source "$PROJECT_DIR/venv/bin/activate"

echo "✅ Virtual environment activated"
echo ""

# Test imports
echo "Testing imports..."
$VENV_PYTHON -c "
try:
    import pandas as pd
    import numpy as np
    import joblib
    from src.config import CLF_STATE_PATH
    from src.preprocessing import preprocess
    from src.decision_engine import decide_what
    print('✅ All core imports successful')
except Exception as e:
    print(f'❌ Import failed: {e}')
    exit(1)
" || exit 1

echo ""
echo "======================================"
echo "🎯 Choose how to run:"
echo "======================================"
echo ""
echo "1️⃣  CLI - Batch predictions"
echo "   python run_inference.py --input data/arvyax_test_inputs_120.xlsx\\ -\\ Sheet1.csv"
echo ""
echo "2️⃣  CLI - Single prediction (interactive)"
echo "   python run_inference.py --interactive"
echo ""
echo "3️⃣  CLI - Single prediction (command line)"
echo "   python run_inference.py --text 'The forest was peaceful' --energy 4 --stress 2"
echo ""
echo "4️⃣  FastAPI Server"
echo "   uvicorn api.main:app --reload"
echo "   Then visit: http://localhost:8000/docs"
echo ""
echo "5️⃣  Gradio Web UI"
echo "   python ui/app.py"
echo "   Then visit: http://localhost:7860"
echo ""
echo "======================================"
echo ""

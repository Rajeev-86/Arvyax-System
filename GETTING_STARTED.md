# 🚀 Getting Started with ArvyaX

## Installation Status

Dependencies are installing in background (this may take 2-5 minutes depending on your internet).

To check progress:
```bash
cd /home/rajeev/Projects/Arvyax_Assignment
source venv/bin/activate
python -c "import pandas; print('✅ Dependencies installed!')"
```

## 🎯 Three Ways to Run ArvyaX

### Option 1: CLI Tool (Fastest)

**Single Prediction (Interactive Mode)**
```bash
cd /home/rajeev/Projects/Arvyax_Assignment
source venv/bin/activate
python run_inference.py --interactive
```

You'll be prompted for:
- Journal text (required)
- Session settings (duration, energy, stress, etc.)
- Whether to generate supportive messages

**Single Prediction (Command Line)**
```bash
python run_inference.py \
  --text "The forest session was peaceful and calming" \
  --ambience forest \
  --duration 20 \
  --energy 4 \
  --stress 2 \
  --time afternoon \
  --quality clear \
  --with-messages
```

**Batch Predictions from CSV**
```bash
python run_inference.py \
  --input data/arvyax_test_inputs_120.xlsx\ -\ Sheet1.csv \
  --output outputs/my_predictions.csv \
  --with-messages
```

### Option 2: Gradio Web UI (Most User-Friendly)

```bash
source venv/bin/activate
python ui/app.py
```

Then open browser: **http://localhost:7860**

Features:
- Interactive form with sliders
- Real-time predictions
- 3 pre-loaded examples
- State probability distribution
- Automatic confidence scoring
- Optional message generation

### Option 3: FastAPI Server (Most Powerful)

```bash
source venv/bin/activate
uvicorn api.main:app --reload
```

Then:
- **Interactive docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

**Example API call (curl):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "journal_text": "The forest was peaceful",
    "ambience_type": "forest",
    "duration_min": 20,
    "energy_level": 4,
    "stress_level": 2,
    "time_of_day": "afternoon",
    "reflection_quality": "clear",
    "generate_message": true
  }'
```

**Example API call (Python):**
```python
import requests

response = requests.post('http://localhost:8000/predict', json={
    "journal_text": "The forest was peaceful",
    "ambience_type": "forest",
    "duration_min": 20,
    "energy_level": 4,
    "stress_level": 2,
    "time_of_day": "afternoon",
    "reflection_quality": "clear",
    "generate_message": True
})

result = response.json()
print(f"State: {result['predicted_state']}")
print(f"What: {result['what_to_do']}")
print(f"When: {result['when_to_do']}")
```

---

## 💻 Python API (Programmatic Use)

```python
import sys
sys.path.insert(0, '/home/rajeev/Projects/Arvyax_Assignment')

from src.inference import EmotionalInferencePipeline

# Initialize pipeline
pipeline = EmotionalInferencePipeline(
    use_slm=False,  # Set to True for Gemma 3 messages (slower)
    use_template_fallback=True
)

# Single prediction
result = pipeline.predict_single({
    'journal_text': 'I felt calm and focused during the session',
    'ambience_type': 'forest',
    'duration_min': 30,
    'sleep_hours': 7.5,
    'energy_level': 4,
    'stress_level': 2,
    'time_of_day': 'afternoon',
    'previous_day_mood': 'calm',
    'face_emotion_hint': None,
    'reflection_quality': 'clear',
}, generate_message=True)

print(f"Emotional State: {result['predicted_state']}")
print(f"Intensity: {result['predicted_intensity']}/5")
print(f"Confidence: {result['confidence']:.1%}")
print(f"What to do: {result['what_to_do']}")
print(f"When to do it: {result['when_to_do']}")
print(f"Message:\n{result['message']}")

# Batch predictions
import pandas as pd

# Load CSV
df = pd.read_csv('data/arvyax_test_inputs_120.xlsx - Sheet1.csv')

# Predict
predictions_df = pipeline.predict_batch(df, generate_messages=True)

# Save
predictions_df.to_csv('outputs/predictions.csv', index=False)
print(f"✅ Saved {len(predictions_df)} predictions")
```

---

## 🧪 Test Without Full Installation

If you want to test basic functionality before all dependencies install:

```bash
cd /home/rajeev/Projects/Arvyax_Assignment

# Test Python syntax (doesn't require dependencies)
python3 -m py_compile src/*.py api/*.py ui/*.py
echo "✅ All Python files are syntactically correct"

# Test imports of our modules (after dependencies install)
python test_system.py
```

---

## 📊 Expected Output Example

**CLI Interactive Mode:**
```
🌿 ArvyaX - Interactive Prediction Mode
============================================================

📝 Journal text: The forest session was really peaceful
🌲 Ambience (forest/ocean/mountain/rain/cafe): forest
⏱️  Duration (minutes): 20
🛏️  Sleep hours (optional): 7.5
⚡ Energy (1-5): 4
😰 Stress (1-5): 2
🕐 Time of day (morning/afternoon/evening/night): afternoon
✨ Reflection quality (clear/vague/conflicted): clear
💬 Generate message? (y/n): y

⏳ Running prediction...

============================================================
🎯 PREDICTION RESULT
============================================================
Emotional State: CALM
Intensity: 3/5
Confidence: 75%
Uncertain: No ✅

📋 Recommendation:
  What: Deep Work
  When: Within 15 Minutes

💬 Message:
  You're in a great mental state for focused work. Channel this calm into your most important task.
============================================================
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"

**Solution:** Dependencies are still installing. Wait a moment and try:
```bash
source venv/bin/activate && python -m pip install --upgrade pandas
```

### "CUDA not available" warnings

**That's OK!** The system works perfectly on CPU. GPU is optional for speedup.

### "SentenceTransformer downloading model..."

**That's normal.** First run downloads embedding model (~90MB). This happens automatically, takes 1-2 minutes.

### Port already in use (8000, 7860)

**Solution:** Change port:
```bash
# FastAPI (default 8000)
uvicorn api.main:app --host 0.0.0.0 --port 8001

# Gradio (default 7860)
python ui/app.py --share  # or change port in code
```

### Low memory warnings

**That's OK!** System is memory-efficient. If you get OOM errors:
```bash
# Run with threading (lower memory)
python run_inference.py --input data.csv
```

---

## 📈 What's Next?

1. **Explore predictions.csv** - See what the model generates
2. **Read ERROR_ANALYSIS.md** - Understand failure modes
3. **Read EDGE_PLAN.md** - See deployment options
4. **Try custom inputs** - Test with your own reflections
5. **Customize rules** - Edit `src/decision_engine.py` WHAT_RULES and WHEN_RULES

---

## 📚 Key Files to Know

| File | Purpose | Try this command |
|------|---------|-----------------|
| `run_inference.py` | CLI tool | `python run_inference.py --help` |
| `ui/app.py` | Web UI | `python ui/app.py` |
| `api/main.py` | API server | `uvicorn api.main:app --reload` |
| `src/inference.py` | Core pipeline | `python -c "from src.inference import EmotionalInferencePipeline"` |
| `README.md` | Full docs | Read in editor |
| `ERROR_ANALYSIS.md` | Failure analysis | Read failure cases |

---

## ✅ Verification Checklist

Once dependencies finish installing:

- [ ] Run: `python test_system.py` (tests all modules)
- [ ] Run: `python run_inference.py --interactive` (quick test)
- [ ] Run: `python ui/app.py` (open in browser)
- [ ] Run: `uvicorn api.main:app --reload` (API docs)

---

## 🎯 Common Questions

**Q: How long does first prediction take?**
A: ~30 seconds (first run downloads embedder model). After that, ~2 seconds per prediction on CPU.

**Q: Can I use GPU?**
A: Yes! If you have CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Q: Does it work offline?**
A: After first run, yes! The embedding model is cached locally.

**Q: Can I customize the messages?**
A: Yes! Edit `src/message_generator.py` MESSAGE_TEMPLATES dict.

**Q: How accurate is the model?**
A: F1=0.586 on training data. See README.md and ERROR_ANALYSIS.md for details.

---

## 🚀 Ready? Start with:

```bash
cd /home/rajeev/Projects/Arvyax_Assignment

# Activate environment
source venv/bin/activate

# Run quickstart guide
bash quickstart.sh

# OR try interactive prediction
python run_inference.py --interactive

# OR start web UI
python ui/app.py
```

---

**Need help?** Check:
- README.md → Full documentation
- ERROR_ANALYSIS.md → Known issues
- EDGE_PLAN.md → Deployments
- IMPLEMENTATION_COMPLETE.md → Feature summary

# ArvyaX Emotional Intelligence System

> **Understanding humans → Reasoning under uncertainty → Guiding them toward better mental states**

A machine learning system that predicts emotional state and intensity from user reflections, then provides actionable recommendations for meaningful next steps.

## 🎯 Project Overview

ArvyaX is designed to handle real-world, messy emotional data:

- **📝 Noisy Text:** Users write short, vague, sometimes contradictory reflections
- **📊 Contextual Signals:** Sleep, stress, energy, time of day, past mood
- **🤔 Uncertainty Awareness:** The system knows when it's unsure
- **💡 Actionable Decisions:** Not just labels, but "what to do" and "when to do it"

### Dataset
- **Training:** 1,200 records from immersive sessions (forest, ocean, rain, mountain, café)
- **Test:** 120 records for evaluation
- **Features:** 13 columns including journal text, contextual metadata, emotional state, and intensity

### Key Metrics
- **Emotional State Classes:** calm, focused, mixed, neutral, overwhelmed, restless (6 classes)
- **Intensity:** 1-5 ordinal scale (treated as regression)
- **Evaluation:** Classification accuracy, MAE, confidence calibration, decision quality

### NoteBook
- [launch Colab](https://drive.google.com/file/d/1IqmaTAlL4_PgtWsT8_feGy9pICzIgjdV/view?usp=sharing)

---

## ⚡ Quick Start

### 1. Installation

```bash
# Clone/navigate to project
cd /home/rajeev/Projects/Arvyax_Assignment

# Install dependencies
pip install -r requirements.txt

# Optional: For Gemma 3 SLM support
huggingface-cli login  # Add your HF token
```

### 2. Run Predictions

#### CLI Tool
```bash
# Batch predictions from CSV
python run_inference.py --input data/arvyax_test_inputs_120.xlsx\ -\ Sheet1.csv --output outputs/test_predictions.csv --with-messages

# Single prediction
python run_inference.py \
  --text "The forest session was peaceful, I felt focused" \
  --energy 4 --stress 2 --duration 20 \
  --with-messages

# Interactive mode
python run_inference.py --interactive
```

#### Python API
```python
from src.inference import EmotionalInferencePipeline

pipeline = EmotionalInferencePipeline(use_slm=False)

result = pipeline.predict_single({
    'journal_text': 'I felt calm in the forest',
    'ambience_type': 'forest',
    'duration_min': 20,
    'energy_level': 4,
    'stress_level': 2,
    'time_of_day': 'afternoon',
    'reflection_quality': 'clear',
    'generate_message': True
})

print(result['predicted_state'])      # 'calm'
print(result['what_to_do'])           # 'deep_work'
print(result['when_to_do'])           # 'within_15_min'
print(result['message'])              # Supportive message
```

### 3. Web Interfaces

#### FastAPI Backend
```bash
uvicorn api.main:app --reload
# Docs: http://localhost:8000/docs
```

#### Gradio UI
```bash
python ui/app.py
# Open: http://localhost:7860
```

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────┐
│         Input: Journal + Context        │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  Preprocessing      │
        │ • Text cleaning     │
        │ • Missing impute    │
        │ • Feature eng       │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Feature Engineering│
        │ • Text embeddings   │
        │ • Categorical enc   │
        │ • Feature scaling   │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │   ML Models         │
        │ • State: XGBClass   │
        │ • Intensity: XGBReg │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │ Decision Engine     │
        │ • What (action)     │
        │ • When (timing)     │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Message Generator  │
        │ • Template-based    │
        │ • Optional: SLM     │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │  Output (JSON/CSV)  │
        └─────────────────────┘
```

### Feature Engineering

**Text Features (384-dim):**
- Sentence-transformers embedding (`all-MiniLM-L6-v2`)
- Semantic understanding of journal text
- **Contribution:** 87% of model importance

**Metadata Features (14 columns):**
- Duration, sleep, energy, stress (numeric)
- Stress × Energy (interaction)
- Sleep deficit flag
- Text length, word count, is_short_text
- Categorical: ambience, time_of_day, mood, face_emotion, reflection_quality
- **Contribution:** 13% of model importance

**Total:** 398-dimensional feature vector

### Models

#### Emotional State (Classification)
- **Model:** XGBoost Classifier (300 trees, max_depth=6)
- **Training Accuracy:** 100% (overfit on training data—expected with small dataset)
- **Cross-validation F1:** 0.58 macro
- **Decision:** Probabilistic output → entropy-based confidence

#### Intensity (Regression vs Classification)
- **Model:** XGBoost Regressor (500 trees, max_depth=5)
- **Training MAE:** 0.19 (raw), 0.08 (rounded)
- **Training R²:** 0.96
- **Why Regression?** Ordinal nature (1→5) is preserved; treats intensity as continuous
- **Performance:** Significantly better than classification approach (F1=0.20)

---

## 🧪 Ablation Study

Evaluated 4 configurations on emotional state prediction:

| Model | F1 Score (macro) | Note |
|-------|------------------|------|
| A: TF-IDF only | 0.5846 | Classical NLP baseline |
| B: MiniLM only | 0.5630 | Deep embeddings alone |
| **C: MiniLM + Metadata** | **0.5807** | symentic but suboptimal |
| D: TF-IDF + Metadata | 0.5888 | **Final model** |

**Conclusion:** "Initial tests showed TF-IDF outperforming dense transformer embeddings (MiniLM). However, combining unrestricted TF-IDF with metadata resulted in feature drowning, where the sparse text matrix overshadowed the valuable context signals. By artificially constraining the TF-IDF vocabulary, I was able to prove that metadata does hold predictive power (Model D > Model A), highlighting the delicate balance between text richness and contextual metadata.

---

## 🤔 Decision Logic

### WHAT Rules (State + Intensity + Stress → Action)

Actions recommended:
```
- box_breathing: For high-intensity anxious/stressed states
- yoga: Medium intensity stress relief
- journaling: Processing sadness or confusion
- sound_therapy: Gentle mood lifting
- deep_work: When calm and focused
- movement: For restless, high-energy states
- rest: For tiredness/fatigue
- light_planning: For calm but slightly stressed
- grounding: Anxiety management
- pause: Default/uncertain cases
```

### WHEN Rules (Time + Intensity + Action Type → Timing)

Timing options:
```
- now: High intensity or morning alarm states
- within_15_min: Medium intensity or neutral times
- later_today: Low intensity, afternoon slumps
- tonight: Evening actions or rest recommendations
- tomorrow_morning: Future planning
```

**Example:**
- State: "overwhelmed", Intensity: 4, Stress: 5, Time: morning
- Mapping: (stressed, high, any) → "box_breathing"
- Timing: (morning, high, calming) → "now"
- **Result:** "Do box breathing now"

---

## 📊 Model Uncertainty

### Confidence Scoring

Uses **entropy-based normalization:**

```
confidence = 1 - H(p) / log(n_classes)
```

Where:
- `H(p)` = Shannon entropy of class probabilities
- `n_classes` = 6 emotional states
- Result: [0, 1] with 1 = very confident

### Uncertainty Flag

```
uncertain_flag = 1  if confidence < 0.55  else 0
```

**Why entropy-based?**
- Max probability alone is misleading (can be high even for non-peaked distributions)
- Entropy captures overall distribution peakedness
- Normalized to [0,1] scale for fair comparison

**Test Set Stats:**
- Average confidence: 0.39 (high uncertainty overall)
- Uncertain predictions: 75% of test set
- Interpretation: Model acknowledges difficulty with noisy test data

---

## ⚠️ Key Challenges & Solutions

### 1. Very Short Text ("ok", "fine", "hmm")

**Challenge:** 115 training samples had ≤3 words; hard to interpret

**Solution:**
- Added `is_short_text` feature flag
- Rely more on contextual signals (stress, energy, sleep)
- Use confidence/uncertainty flags to mark these predictions

### 2. Conflicting Signals

**Example:** "I feel calm" but stress=5, sleep=4

**Solution:**
- Decision engine explicitly considers stress AND intensity
- High confidence only when signals align
- Supports multiple legitimate emotional states

### 3. Ambiguous Labels

**Challenge:** Training labels may have noise (ambiguity between "calm" vs "neutral")

**Solution:**
- Class-agnostic metrics (F1-macro, MAE)
- Confidence scores help identify ambiguous cases
- Error analysis (see ERROR_ANALYSIS.md) for label quality assessment

### 4. Imbalanced Distribution

Not a major issue here (classes: 153-216 samples), but handled via:
- Stratified cross-validation
- Weighted metrics (F1-macro)

---

## 📈 Feature Importance (SHAP)

Top 5 most important features:

| Feature | SHAP Importance | Type |
|---------|-----------------|------|
| EMBED_000-EMBED_100 | ~60% | Text embeddings (aggregate) |
| duration_min | 0.1089 | Session length |
| previous_day_mood_enc | 0.1087 | Yesterday's state |
| reflection_quality_enc | 0.1004 | Clarity of reflection |
| sleep_hours | 0.0984 | Sleep quality |

**Insight:** Text is king (87%), but contextual metadata provides crucial signals (13%).

---

## 🔄 How to Use Output

### predictions.csv Structure

```csv
id,predicted_state,predicted_intensity,confidence,uncertain_flag,what_to_do,when_to_do
10001,focused,3,0.5632,0,pause,tonight
10002,restless,4,0.1691,1,movement,now
10003,calm,3,0.3761,1,light_planning,within_15_min
```

### Using Predictions

1. **Show to User:** "You seem _focused_. Try _pause_ (tonight)"
2. **Log Confidence:** Track which predictions were uncertain
3. **Fine-tune When:** Adjust timing based on actual user behavior
4. **Message as Support:** Optional supportive message for empathy

---

## 🚀 API Usage

### FastAPI Endpoints

#### Single Prediction
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

#### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/test.csv"
```

#### Health Check
```bash
curl http://localhost:8000/health
```

---

## 📁 Project Structure

```
Arvyax_Assignment/
├── src/
│   ├── __init__.py              # Main exports
│   ├── config.py                # Constants & paths
│   ├── preprocessing.py         # Text/data cleaning
│   ├── feature_engineering.py   # Embeddings + matrix
│   ├── decision_engine.py       # What/When rules
│   ├── uncertainty.py           # Confidence scoring
│   ├── inference.py             # Pipeline class
│   └── message_generator.py     # Templates + SLM
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app
│   └── schemas.py               # Pydantic models
├── ui/
│   ├── __init__.py
│   └── app.py                   # Gradio interface
├── models/                      # Saved ML artifacts
│   ├── clf_emotional_state.pkl  # XGBoost classifier
│   ├── reg_intensity.pkl        # XGBoost regressor
│   ├── encoders.pkl             # Label encoders
│   ├── scaler.pkl               # Feature scaler
│   └── state_encoder.pkl        # Class encoder
├── data/                        # Raw datasets
├── outputs/                     # predictions.csv + plots
├── run_inference.py             # CLI entry point
├── requirements.txt
├── README.md
├── ERROR_ANALYSIS.md
├── EDGE_PLAN.md
└── arvyax_training_pipeline.ipynb  # Original notebook
```

---

## 🛠️ Customization

### Adjust Decision Rules

Edit `src/decision_engine.py`:

```python
WHAT_RULES = {
    ('anxious', 'high', 'any'): 'box_breathing',  # Modify this
    ...
}

WHEN_RULES = {
    ('morning', 'high', 'calming'): 'now',  # Or this
    ...
}
```

### Add Custom Messages

Edit `src/message_generator.py`:

```python
MESSAGE_TEMPLATES = {
    ("restless", "box_breathing"): [
        "Your custom message here...",
    ],
}
```

### Use SLM for Messages

```python
pipeline = EmotionalInferencePipeline(
    use_slm=True,  # Enable Gemma 3 270M
    use_template_fallback=True
)
```

---

## 🎓 Model Details

### Training Process

1. **Preprocessing:** 1,200 → 1,200 (no row drops)
2. **Feature Engineering:** 13 columns → 398 features
3. **Train/Val Split:** Stratified K-fold (5 splits)
4. **Hyperparameter Tuning:** Via cross-validation
5. **Model Selection:** Best CV score wins

### Cross-Validation Results

**Emotional State (Classification):**
- Fold 1: F1 = 0.606 ± 0.023
- Fold 2: F1 = 0.581 ± 0.022
- Fold 3: F1 = 0.576 ± 0.020
- Fold 4: F1 = 0.579 ± 0.019
- Fold 5: F1 = 0.589 ± 0.021
- **Average:** 0.586 ± 0.021

**Intensity (Regression):**
- MAE: 1.33 ± 0.05
- R²: 0.88 ± 0.05

### Hyperparameters

```python
# Emotional State Classifier
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss'
)

# Intensity Regressor
XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.8,
    min_child_weight=2,
    reg_alpha=0.1,
    reg_lambda=1.0
)
```

---

## 🔍 Evaluation & Testing

### Metrics Used

- **Classification:** Accuracy, F1 (macro), Precision, Recall, Confusion Matrix
- **Regression:** MAE, MSE, R²
- **Confidence:** Expected Calibration Error (ECE)
- **Decision Quality:** Manual review of actionability

### Test Set Performance

See `predictions.csv` for full results. Summary:

| Metric | Value |
|--------|-------|
| Avg Confidence | 0.39 |
| Uncertain Predictions | 75% |
| Prediction Diversity | 6 states in predictions |
| Action Diversity | 8 different actions |

---

## 🐛 Troubleshooting

### Models not found
```bash
# Ensure models/ directory has all .pkl files
ls models/
```

### Import errors
```bash
# Reinstall from requirements
pip install -r requirements.txt
```

### CUDA errors (for GPU support)
```bash
# Install CPU or GPU-specific PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Gemma not loading
The system gracefully falls back to template-based messages. Check:
```bash
huggingface-cli whoami  # Verify token
```

---

## 📚 References

- **Training Notebook:** `arvyax_training_pipeline.ipynb` - Full EDA, model training, ablation study
- **Error Analysis:** `ERROR_ANALYSIS.md` - Deep dive into 10+ failure cases
- **Edge Deployment:** `EDGE_PLAN.md` - Optimization strategies for mobile/on-device

---

## 📜 License & Authors

Built for ArvyaX Internship Assignment | Team RevoltronX

---

## 🤝 Support

For issues or questions:
1. Check `ERROR_ANALYSIS.md` for known limitations
2. Review `EDGE_PLAN.md` for deployment considerations
3. Consult `arvyax_training_pipeline.ipynb` for training details

---

**Last Updated:** March 2026
**Version:** 1.0.0

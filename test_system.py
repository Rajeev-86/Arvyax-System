#!/usr/bin/env python3
"""
Quick test script for ArvyaX system.
Run this to verify everything is working.
"""

import sys
sys.path.insert(0, '/home/rajeev/Projects/Arvyax_Assignment')

print("🌿 ArvyaX System Test")
print("=" * 60)

# Test 1: Imports
print("\n1️⃣  Testing imports...")
try:
    import pandas as pd
    import numpy as np
    import joblib
    from src.config import CLF_STATE_PATH, N_CLASSES, EMOTIONAL_STATES
    from src.preprocessing import preprocess_single
    from src.decision_engine import decide_what, decide_when
    from src.uncertainty import compute_confidence_single
    from src.inference import EmotionalInferencePipeline
    print("   ✅ All imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Config
print("\n2️⃣  Checking configuration...")
try:
    assert N_CLASSES == 6, f"Expected 6 classes, got {N_CLASSES}"
    assert 'calm' in EMOTIONAL_STATES, "Missing 'calm' state"
    print(f"   ✅ Config OK: {N_CLASSES} classes, {len(EMOTIONAL_STATES)} states")
except Exception as e:
    print(f"   ❌ Config check failed: {e}")
    sys.exit(1)

# Test 3: Models loaded
print("\n3️⃣  Checking models...")
try:
    assert CLF_STATE_PATH.exists(), f"Classifier not found at {CLF_STATE_PATH}"
    clf = joblib.load(CLF_STATE_PATH)
    print(f"   ✅ Models found and loadable")
except Exception as e:
    print(f"   ❌ Model check failed: {e}")
    sys.exit(1)

# Test 4: Decision engine
print("\n4️⃣  Testing decision engine...")
try:
    what = decide_what('calm', 3, 2)
    when = decide_when('afternoon', 3, what)
    assert isinstance(what, str) and len(what) > 0
    assert isinstance(when, str) and len(when) > 0
    print(f"   ✅ Decision engine OK: {what} → {when}")
except Exception as e:
    print(f"   ❌ Decision engine test failed: {e}")
    sys.exit(1)

# Test 5: Preprocessing
print("\n5️⃣  Testing preprocessing...")
try:
    test_input = {
        'journal_text': 'The forest was peaceful',
        'stress_level': 2,
        'energy_level': 4
    }
    processed = preprocess_single(test_input)
    assert 'journal_text' in processed
    assert 'text_length' in processed
    assert 'word_count' in processed
    print(f"   ✅ Preprocessing OK: '{test_input['journal_text']}' → {processed['word_count']} words")
except Exception as e:
    print(f"   ❌ Preprocessing test failed: {e}")
    sys.exit(1)

# Test 6: Pipeline initialization (fast version - no SLM)
print("\n6️⃣  Testing pipeline initialization...")
try:
    print("   ⏳ Loading models (this takes 30-60s first time)...")
    pipeline = EmotionalInferencePipeline(use_slm=False, use_template_fallback=True)
    print("   ✅ Pipeline initialized")
except Exception as e:
    print(f"   ❌ Pipeline init failed: {e}")
    print("   💡 This is OK - it means embedding model needs to download (~90MB)")
    sys.exit(1)

# Test 7: Single prediction
print("\n7️⃣  Testing single prediction...")
try:
    test_data = {
        'journal_text': 'The forest session was peaceful and calm',
        'ambience_type': 'forest',
        'duration_min': 20,
        'sleep_hours': 7.5,
        'energy_level': 4,
        'stress_level': 2,
        'time_of_day': 'afternoon',
        'previous_day_mood': 'calm',
        'face_emotion_hint': None,
        'reflection_quality': 'clear',
    }
    result = pipeline.predict_single(test_data, generate_message=False)

    print(f"   ✅ Prediction successful:")
    print(f"      State: {result['predicted_state']}")
    print(f"      Intensity: {result['predicted_intensity']}/5")
    print(f"      Confidence: {result['confidence']:.2%}")
    print(f"      What to do: {result['what_to_do']}")
    print(f"      When: {result['when_to_do']}")
except Exception as e:
    print(f"   ❌ Prediction test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\n🎯 Next steps:")
print("   1. Run: python run_inference.py --interactive")
print("   2. Or: python ui/app.py")
print("   3. Or: uvicorn api.main:app --reload")

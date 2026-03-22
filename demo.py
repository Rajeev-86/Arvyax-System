#!/usr/bin/env python3
"""
Simple demo script to test ArvyaX system with a few sample predictions.
Run once dependencies are installed.
"""

import sys
sys.path.insert(0, '/home/rajeev/Projects/Arvyax_Assignment')

def main():
    print("\n" + "="*70)
    print("🌿 ArvyaX Emotional Intelligence System - Demo")
    print("="*70 + "\n")

    # Import
    print("Loading system...")
    from src.inference import EmotionalInferencePipeline

    pipeline = EmotionalInferencePipeline(use_slm=False)
    print("✅ System loaded\n")

    # Sample test cases
    test_cases = [
        {
            "name": "Peaceful Forest Session",
            "data": {
                'journal_text': 'The forest session was peaceful and calming. I felt focused.',
                'ambience_type': 'forest',
                'duration_min': 30,
                'sleep_hours': 8.0,
                'energy_level': 4,
                'stress_level': 2,
                'time_of_day': 'afternoon',
                'reflection_quality': 'clear',
            }
        },
        {
            "name": "Restless Ocean Session",
            "data": {
                'journal_text': 'Felt restless and agitated the whole time.',
                'ambience_type': 'ocean',
                'duration_min': 15,
                'sleep_hours': 5.0,
                'energy_level': 2,
                'stress_level': 4,
                'time_of_day': 'night',
                'reflection_quality': 'vague',
            }
        },
        {
            "name": "Mixed Emotions Mountain Session",
            "data": {
                'journal_text': 'Felt calm but also a bit anxious about work.',
                'ambience_type': 'mountain',
                'duration_min': 45,
                'sleep_hours': 7.0,
                'energy_level': 3,
                'stress_level': 3,
                'time_of_day': 'morning',
                'reflection_quality': 'clear',
            }
        },
        {
            "name": "Overwhelmed Rainy Session",
            "data": {
                'journal_text': 'Too much on my mind. Feeling overwhelmed.',
                'ambience_type': 'rain',
                'duration_min': 10,
                'sleep_hours': 4.5,
                'energy_level': 1,
                'stress_level': 5,
                'time_of_day': 'evening',
                'reflection_quality': 'conflicted',
            }
        },
    ]

    # Run predictions
    for i, test in enumerate(test_cases, 1):
        print(f"{i}) {test['name']}")
        print("-" * 70)

        result = pipeline.predict_single(test['data'], generate_message=True)

        print(f"   📝 Text: \"{test['data']['journal_text'][:50]}...\"")
        print(f"   🎯 Predicted State: {result['predicted_state'].upper()}")
        print(f"   📊 Intensity: {result['predicted_intensity']}/5")
        print(f"   🤔 Confidence: {result['confidence']:.1%}")

        if result['uncertain_flag']:
            print(f"   ⚠️  Uncertain prediction")
        else:
            print(f"   ✅ Confident prediction")

        print(f"\n   💡 Recommendation:")
        print(f"      What: {result['what_to_do'].replace('_', ' ').title()}")
        print(f"      When: {result['when_to_do'].replace('_', ' ').title()}")

        if result['message']:
            msg = result['message']
            # Word wrap message to 60 chars
            import textwrap
            wrapped = textwrap.fill(msg, width=60)
            print(f"\n   💬 Message:")
            for line in wrapped.split('\n'):
                print(f"      {line}")

        print()

    print("="*70)
    print("✅ Demo complete!")
    print("="*70)
    print("\n🎯 What to do next:")
    print("   1. Try interactive mode:  python run_inference.py --interactive")
    print("   2. Try web UI:            python ui/app.py")
    print("   3. Try API server:        uvicorn api.main:app --reload")
    print()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

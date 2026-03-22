#!/usr/bin/env python
"""CLI tool for batch and single predictions."""

import argparse
import sys
import json
from pathlib import Path

import pandas as pd

from src.inference import EmotionalInferencePipeline


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ArvyaX Emotional Intelligence - CLI Prediction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python run_inference.py \\
    --text "The forest was peaceful" \\
    --energy 4 --stress 2 --duration 20

  # Batch from CSV
  python run_inference.py \\
    --input data/test.csv \\
    --output predictions.csv \\
    --with-messages

  # Interactive mode
  python run_inference.py --interactive
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Input CSV file for batch predictions"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions.csv",
        help="Output CSV file path (default: outputs/predictions.csv)"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Journal text for single prediction"
    )
    parser.add_argument(
        "--ambience",
        type=str,
        default="forest",
        help="Ambience type: forest, ocean, mountain, rain, cafe"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=15,
        help="Session duration in minutes"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        help="Hours slept (optional)"
    )
    parser.add_argument(
        "--energy",
        type=int,
        default=3,
        help="Energy level 1-5"
    )
    parser.add_argument(
        "--stress",
        type=int,
        default=3,
        help="Stress level 1-5"
    )
    parser.add_argument(
        "--time",
        type=str,
        default="afternoon",
        help="Time of day: morning, afternoon, evening, night"
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="clear",
        help="Reflection quality: clear, vague, conflicted"
    )
    parser.add_argument(
        "--with-messages",
        action="store_true",
        help="Generate supportive messages"
    )
    parser.add_argument(
        "--with-slm",
        action="store_true",
        help="Use SLM for message generation (if available)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode (prompt for inputs)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )

    args = parser.parse_args()

    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = EmotionalInferencePipeline(
        use_slm=args.with_slm,
        use_template_fallback=True
    )

    # Batch prediction from CSV
    if args.input:
        print(f"Loading input file: {args.input}")
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} records")

        print("Running predictions...")
        results = pipeline.predict_batch(df, generate_messages=args.with_messages)

        # Save output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to: {output_path}")
        print(f"\nSample predictions:")
        print(results.head(3).to_string())

    # Single prediction from command line
    elif args.text:
        input_data = {
            'journal_text': args.text,
            'ambience_type': args.ambience,
            'duration_min': args.duration,
            'sleep_hours': args.sleep,
            'energy_level': args.energy,
            'stress_level': args.stress,
            'time_of_day': args.time,
            'previous_day_mood': None,
            'face_emotion_hint': None,
            'reflection_quality': args.quality,
        }

        print("Running prediction...")
        result = pipeline.predict_single(input_data, generate_message=args.with_messages)

        # Output
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print("\n" + "=" * 60)
            print("🎯 PREDICTION RESULT")
            print("=" * 60)
            print(f"Emotional State: {result['predicted_state'].upper()}")
            print(f"Intensity: {result['predicted_intensity']}/5")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Uncertain: {'Yes ⚠️' if result['uncertain_flag'] else 'No ✅'}")
            print(f"\n📋 Recommendation:")
            print(f"  What: {result['what_to_do'].replace('_', ' ').title()}")
            print(f"  When: {result['when_to_do'].replace('_', ' ').title()}")
            if result['message']:
                print(f"\n💬 Message:\n  {result['message']}")
            print("=" * 60)

    # Interactive mode
    elif args.interactive:
        print("\n🌿 ArvyaX - Interactive Prediction Mode")
        print("=" * 60)

        journal_text = input("\n📝 Journal text: ").strip()
        if not journal_text:
            print("❌ Journal text is required")
            return

        ambience = input("🌲 Ambience (forest/ocean/mountain/rain/cafe): ").strip() or "forest"
        duration = int(input("⏱️  Duration (minutes): ") or "15")
        sleep = input("🛏️  Sleep hours (optional): ")
        energy = int(input("⚡ Energy (1-5): ") or "3")
        stress = int(input("😰 Stress (1-5): ") or "3")
        time_of_day = input("🕐 Time of day (morning/afternoon/evening/night): ").strip() or "afternoon"
        quality = input("✨ Reflection quality (clear/vague/conflicted): ").strip() or "clear"
        gen_msg = input("💬 Generate message? (y/n): ").lower() == "y"

        input_data = {
            'journal_text': journal_text,
            'ambience_type': ambience,
            'duration_min': duration,
            'sleep_hours': float(sleep) if sleep else None,
            'energy_level': energy,
            'stress_level': stress,
            'time_of_day': time_of_day,
            'previous_day_mood': None,
            'face_emotion_hint': None,
            'reflection_quality': quality,
        }

        print("\n⏳ Running prediction...")
        result = pipeline.predict_single(input_data, generate_message=gen_msg)

        print("\n" + "=" * 60)
        print("🎯 PREDICTION RESULT")
        print("=" * 60)
        print(f"Emotional State: {result['predicted_state'].upper()}")
        print(f"Intensity: {result['predicted_intensity']}/5")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Uncertain: {'Yes ⚠️' if result['uncertain_flag'] else 'No ✅'}")
        print(f"\n📋 Recommendation:")
        print(f"  What: {result['what_to_do'].replace('_', ' ').title()}")
        print(f"  When: {result['when_to_do'].replace('_', ' ').title()}")
        if result['message']:
            print(f"\n💬 Message:\n  {result['message']}")
        print("=" * 60)

    else:
        print("❌ Please provide either --input, --text, or --interactive")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

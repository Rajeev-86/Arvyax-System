"""Gradio web interface for ArvyaX emotional intelligence system."""

import sys
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import EmotionalInferencePipeline
from src.config import EMOTIONAL_STATES

# Initialize pipeline
print("Initializing pipeline for UI...")
pipeline = EmotionalInferencePipeline(use_slm=False, use_template_fallback=True)


def predict_emotion(
    journal_text: str,
    ambience_type: str,
    duration_min: int,
    sleep_hours: float,
    energy_level: int,
    stress_level: int,
    time_of_day: str,
    previous_day_mood: str,
    face_emotion_hint: str,
    reflection_quality: str,
    generate_message: bool
) -> tuple:
    """Make prediction and return formatted results."""

    try:
        # Prepare input
        input_data = {
            'journal_text': journal_text,
            'ambience_type': ambience_type,
            'duration_min': int(duration_min),
            'sleep_hours': float(sleep_hours) if sleep_hours else None,
            'energy_level': int(energy_level),
            'stress_level': int(stress_level),
            'time_of_day': time_of_day,
            'previous_day_mood': previous_day_mood if previous_day_mood else None,
            'face_emotion_hint': face_emotion_hint if face_emotion_hint else None,
            'reflection_quality': reflection_quality,
        }

        # Get prediction
        result = pipeline.predict_single(input_data, generate_message=generate_message)

        # Format output
        output_text = f"""
**📊 Emotional State Analysis**

🎯 **Predicted State:** {result['predicted_state'].upper()}
📈 **Intensity:** {result['predicted_intensity']}/5
🤔 **Confidence:** {result['confidence']:.1%}
⚠️ **Uncertain:** {"Yes ⚠️" if result['uncertain_flag'] else "No ✅"}

**💡 Recommendation**

📋 **What to do:** {result['what_to_do'].replace('_', ' ').title()}
⏰ **When to do it:** {result['when_to_do'].replace('_', ' ').title()}
"""

        if result['message']:
            output_text += f"\n💬 **Supportive Message:**\n\n_{result['message']}_\n"

        # State probabilities
        prob_text = "**📊 State Probabilities:**\n\n"
        for state, prob in sorted(
            result['state_probabilities'].items(),
            key=lambda x: -x[1]
        ):
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            prob_text += f"{state:12} {bar} {prob:.1%}\n"

        return output_text, prob_text

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, "Error occurred"


# Create Gradio interface
with gr.Blocks(
    title="ArvyaX - Emotional Intelligence Guide",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown("# 🌿 ArvyaX Emotional Intelligence System")
    gr.Markdown(
        """
        **Understand your emotions → Get meaningful recommendations → Guide yourself to better states**

        Fill in your reflections and context, and receive personalized guidance.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Your Reflection")

            journal_text = gr.Textbox(
                label="Journal Entry / Reflection",
                placeholder="Write what you're feeling or what happened in the session...",
                lines=4,
                interactive=True
            )

            gr.Markdown("### Session Context")

            ambience_type = gr.Dropdown(
                choices=["forest", "ocean", "mountain", "rain", "cafe"],
                label="Ambience Type",
                value="forest"
            )

            duration_min = gr.Slider(
                minimum=1,
                maximum=180,
                value=15,
                step=1,
                label="Session Duration (minutes)"
            )

            gr.Markdown("### Current State")

            energy_level = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="Energy Level (1=very low, 5=very high)"
            )

            stress_level = gr.Slider(
                minimum=1,
                maximum=5,
                value=3,
                step=1,
                label="Stress Level (1=calm, 5=very stressed)"
            )

            sleep_hours = gr.Number(
                label="Hours Slept Last Night",
                value=7,
                interactive=True
            )

            time_of_day = gr.Dropdown(
                choices=["early_morning", "morning", "afternoon", "evening", "night"],
                label="Time of Day",
                value="afternoon"
            )

            gr.Markdown("### Additional Context (Optional)")

            previous_day_mood = gr.Dropdown(
                choices=["calm", "mixed", "neutral", "focused", "overwhelmed", ""],
                label="Previous Day Mood",
                value=""
            )

            face_emotion_hint = gr.Dropdown(
                choices=["calm_face", "tired_face", "neutral_face", "happy_face", "tense_face", ""],
                label="Apparent Emotion (if known)",
                value=""
            )

            reflection_quality = gr.Dropdown(
                choices=["clear", "vague", "conflicted"],
                label="How Clear is Your Reflection?",
                value="clear"
            )

            generate_message = gr.Checkbox(
                label="Generate Supportive Message",
                value=True
            )

        with gr.Column(scale=1):
            gr.Markdown("## 🎯 Your Personalized Guidance")

            analysis_output = gr.Markdown(
                value="Your analysis will appear here...",
            )

            probabilities_output = gr.Markdown(
                value="State probability distribution...",
            )

    # Button
    predict_btn = gr.Button("🔮 Analyze My Emotional State", variant="primary", size="lg")

    predict_btn.click(
        fn=predict_emotion,
        inputs=[
            journal_text, ambience_type, duration_min, sleep_hours,
            energy_level, stress_level, time_of_day, previous_day_mood,
            face_emotion_hint, reflection_quality, generate_message
        ],
        outputs=[analysis_output, probabilities_output]
    )

    # Examples
    gr.Examples(
        examples=[
            [
                "The forest was peaceful. I felt calm and thoughtful the whole time.",
                "forest", 20, 8, 4, 2, "afternoon", "calm", "calm_face", "clear", True
            ],
            [
                "I was restless the whole session. Couldn't focus.",
                "ocean", 15, 6, 2, 4, "morning", "mixed", "tense_face", "vague", True
            ],
            [
                "Overwhelmed. Too much happening at once.",
                "mountain", 30, 5, 1, 5, "evening", "overwhelmed", "", "conflicted", True
            ],
        ],
        inputs=[
            journal_text, ambience_type, duration_min, sleep_hours,
            energy_level, stress_level, time_of_day, previous_day_mood,
            face_emotion_hint, reflection_quality, generate_message
        ],
        outputs=[analysis_output, probabilities_output],
        fn=predict_emotion,
        cache_examples=False,
        label="📚 Try These Examples"
    )

    gr.Markdown(
        """
        ---

        ### 🔍 How It Works

        1. **Your Input:** Journal text + contextual signals (sleep, stress, energy, etc.)
        2. **Our Analysis:** Advanced ML models analyze both text and context
        3. **Your Guidance:**
           - Predicted emotional state
           - Intensity and confidence
           - **What to do** (specific recommendation)
           - **When to do it** (timing guidance)
           - Optional supportive message

        ### ⚠️ Important

        This system is designed to *support*, not replace professional mental health care.
        If you're in crisis, please reach out to a mental health professional.
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

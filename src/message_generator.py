"""Message generation - template and optional SLM-based."""

from typing import Optional, Dict
import random

from .config import SLM_MODEL_NAME, SLM_USE_TEMPLATE_FALLBACK


# Template-based messages for edge deployment and fast fallback
MESSAGE_TEMPLATES = {
    # (state, action) -> message template
    ("restless", "box_breathing"): [
        "You seem slightly restless right now. Let's slow things down. Try a short breathing exercise before planning your day.",
        "I sense you have a lot of energy flowing. Channel it wisely with some controlled breathing first.",
        "Restlessness often signals your mind wants direction. A brief pause and some deep breaths can help calm the storm.",
    ],
    ("restless", "movement"): [
        "Your body wants to move. Go for a walk or do some light stretching—it'll help settle your mind.",
        "Channel that restless energy into movement. A short walk or dance session will help.",
    ],
    ("restless", "box_breathing"): [
        "You're feeling agitated. Try box breathing to regulate your nervous system.",
        "Take a moment to pause. Four counts in, hold four, out four—let's reset.",
    ],
    ("overwhelmed", "yoga"): [
        "I sense you're carrying a lot right now. A gentle yoga session can help you release what's weighing on you.",
        "Overwhelm often comes from holding tension. Yoga can help you breathe through it.",
    ],
    ("overwhelmed", "journaling"): [
        "Writing might help untangle what's overwhelming you. Let your thoughts flow onto the page.",
        "Sometimes putting feelings into words helps. Try journaling what's on your mind.",
    ],
    ("overwhelmed", "box_breathing"): [
        "Feeling swamped? Let's focus on just the next breath. Box breathing can steady you.",
        "Take control of what you can—your breath. Try box breathing to calm the overwhelm.",
    ],
    ("calm", "deep_work"): [
        "You're in a great mental state for focused work. Channel this calm into your most important task.",
        "Your mind is settled. This is prime time for deep, meaningful work.",
        "You're calm and centered. Perfect conditions for deep focus. Let's get to that important work.",
    ],
    ("calm", "light_planning"): [
        "You're centered right now. This is a good time to plan your next steps thoughtfully.",
        "With this calm state, you can plan clearly. What's next for you?",
    ],
    ("focused", "deep_work"): [
        "You're already in the zone. Keep this momentum going with focused work on what matters most.",
        "Your mind is sharp and focused. Time to dive deep into important work.",
    ],
    ("focused", "light_planning"): [
        "You're clear-headed. Use this focus to plan what comes next.",
        "Your thoughts are organized. Now's the time to structure your day.",
    ],
    ("mixed", "pause"): [
        "Your emotions are mixed right now. Sometimes the best move is to pause and let them settle before deciding.",
        "Mixed feelings are normal. Give yourself permission to just sit with them for a moment.",
    ],
    ("mixed", "journaling"): [
        "Your emotions are mixing. Writing can help you sort through what you're feeling.",
        "Multiple feelings happening at once? Let's journal through them.",
    ],
    ("neutral", "light_planning"): [
        "You're in a balanced state. This is a good time for practical planning and organization.",
        "A neutral state is perfect for thinking clearly. What needs your attention today?",
    ],
    ("neutral", "pause"): [
        "You're feeling neutral. No rush to do anything right now. You can rest or gently plan.",
        "Neutral is fine. You can take your time with the next decision.",
    ],
    ("sad", "journaling"): [
        "I sense some sadness. Writing about how you're feeling can be incredibly healing.",
        "Sadness is valid. Journaling can help you understand what's beneath the surface.",
    ],
    ("sad", "sound_therapy"): [
        "Music can soothe what words can't. Try some calming sounds or music you love.",
        "Let some gentle music or nature sounds wash over you. Healing happens in softness.",
    ],
    ("sad", "rest"): [
        "You're feeling low. Give yourself permission to rest and be gentle with yourself.",
        "Sadness sometimes means you need compassion—especially from yourself. Rest awhile.",
    ],
}


class MessageGenerator:
    """Generate supportive messages using templates or optional SLM."""

    def __init__(self, use_slm: bool = False, use_template_fallback: bool = True):
        """
        Initialize message generator.

        Args:
            use_slm: Whether to attempt loading SLM model
            use_template_fallback: Whether to fallback to templates if SLM fails/unavailable
        """
        self.use_slm = use_slm
        self.use_template_fallback = use_template_fallback
        self.slm_model = None
        self.slm_tokenizer = None

        if use_slm:
            self._load_slm()

    def _load_slm(self) -> None:
        """Load Gemma 3 SLM if available."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading SLM: {SLM_MODEL_NAME}...")
            self.slm_tokenizer = AutoTokenizer.from_pretrained(SLM_MODEL_NAME, trust_remote_code=True)
            self.slm_model = AutoModelForCausalLM.from_pretrained(
                SLM_MODEL_NAME,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
            print("✅ SLM loaded successfully")
        except Exception as e:
            print(f"⚠️  Failed to load SLM: {e}")
            print("   Falling back to template-based messages")
            self.slm_model = None
            self.slm_tokenizer = None

    def _generate_with_slm(
        self,
        state: str,
        intensity: int,
        action: str,
        confidence: float,
        uncertain: int
    ) -> str:
        """Generate message using SLM."""
        if self.slm_model is None or self.slm_tokenizer is None:
            return None

        try:
            prompt = f"""You are a supportive AI guide for emotional wellness.
Generate a brief, warm, and actionable message for someone with:
- Emotional state: {state}
- Intensity: {intensity}/5
- Recommended action: {action}
- Confidence in prediction: {confidence:.2f}
- Is uncertain: {'yes' if uncertain else 'no'}

Keep it under 50 words. Be human-like, not robotic."""

            inputs = self.slm_tokenizer(prompt, return_tensors="pt")
            outputs = self.slm_model.generate(
                **inputs,
                max_new_tokens=70,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            message = self.slm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part (remove prompt)
            message = message.replace(prompt, "").strip()
            return message if message else None
        except Exception as e:
            print(f"SLM generation failed: {e}")
            return None

    def generate(
        self,
        state: str,
        intensity: int,
        action: str,
        confidence: float = 0.5,
        uncertain: int = 0
    ) -> str:
        """
        Generate a supportive message.

        Args:
            state: Predicted emotional state
            intensity: Predicted intensity (1-5)
            action: Recommended action
            confidence: Model confidence (0-1)
            uncertain: Whether prediction is uncertain (0 or 1)

        Returns:
            Generated message string, or None if both failed
        """
        # Try SLM first if enabled
        if self.use_slm and self.slm_model is not None:
            message = self._generate_with_slm(state, intensity, action, confidence, uncertain)
            if message:
                return message

        # Fallback to templates
        if self.use_template_fallback:
            return self._get_template_message(state, action, uncertainty_aware=uncertain == 1)

        return None

    def _get_template_message(self, state: str, action: str, uncertainty_aware: bool = False) -> str:
        """Get template-based message."""
        key = (state.lower(), action.lower())

        # Try exact match
        if key in MESSAGE_TEMPLATES:
            messages = MESSAGE_TEMPLATES[key]
            return random.choice(messages)

        # Fallback to generic messages
        if action == 'rest':
            return "You deserve a break. Rest for a bit and come back refreshed."
        elif action == 'deep_work':
            return "You're in a good place to tackle meaningful work right now."
        elif action == 'journaling':
            return "Your feelings matter. Take some time to write them down."
        elif 'breathing' in action:
            return "Let's pause and take some mindful breaths together."
        elif action == 'movement':
            return "Your body needs some movement. Go stretch or take a walk."
        else:
            if uncertainty_aware:
                return "I'm not entirely sure what's going on, but I'm here to listen. Take a moment for yourself."
            return "You've got this. Trust your instincts and take it one step at a time."

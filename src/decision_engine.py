"""Decision engine for determining what to do and when."""

from typing import Tuple

# Action types for WHEN rules
ACTION_TYPE_MAP = {
    'box_breathing': 'calming',
    'grounding': 'calming',
    'journaling': 'calming',
    'sound_therapy': 'calming',
    'yoga': 'calming',
    'rest': 'rest',
    'pause': 'rest',
    'movement': 'energizing',
    'deep_work': 'work',
    'light_planning': 'work',
}

# WHAT rules: (state_group, intensity_bucket, stress_bucket) -> action
WHAT_RULES = {
    ('anxious', 'high', 'any'): 'box_breathing',
    ('anxious', 'medium', 'any'): 'grounding',
    ('anxious', 'low', 'any'): 'grounding',
    ('stressed', 'high', 'any'): 'box_breathing',
    ('stressed', 'medium', 'any'): 'yoga',
    ('stressed', 'low', 'any'): 'light_planning',
    ('sad', 'high', 'any'): 'sound_therapy',
    ('sad', 'medium', 'any'): 'journaling',
    ('sad', 'low', 'any'): 'journaling',
    ('calm', 'any', 'low'): 'deep_work',
    ('calm', 'any', 'high'): 'light_planning',
    ('calm', 'any', 'any'): 'light_planning',
    ('happy', 'any', 'any'): 'deep_work',
    ('excited', 'any', 'any'): 'deep_work',
    ('content', 'any', 'any'): 'deep_work',
    ('tired', 'any', 'any'): 'rest',
    ('fatigued', 'any', 'any'): 'rest',
    ('confused', 'any', 'any'): 'journaling',
    ('restless', 'high', 'any'): 'movement',
    ('restless', 'any', 'any'): 'box_breathing',
    ('default', 'any', 'any'): 'pause',
}

# WHEN rules: (time_group, intensity_bucket, action_type) -> timing
WHEN_RULES = {
    ('morning', 'high', 'calming'): 'now',
    ('morning', 'high', 'any'): 'now',
    ('morning', 'medium', 'calming'): 'within_15_min',
    ('morning', 'medium', 'any'): 'within_15_min',
    ('morning', 'low', 'work'): 'within_15_min',
    ('morning', 'low', 'any'): 'within_15_min',
    ('afternoon', 'high', 'calming'): 'now',
    ('afternoon', 'high', 'any'): 'now',
    ('afternoon', 'medium', 'any'): 'within_15_min',
    ('afternoon', 'low', 'any'): 'later_today',
    ('evening', 'high', 'calming'): 'now',
    ('evening', 'high', 'any'): 'now',
    ('evening', 'any', 'rest'): 'tonight',
    ('evening', 'any', 'any'): 'tonight',
    ('night', 'high', 'calming'): 'now',
    ('night', 'any', 'any'): 'tonight',
    ('default', 'any', 'any'): 'within_15_min',
}


def bucket_intensity(intensity: float) -> str:
    """Convert intensity (1-5) to bucket."""
    i = float(intensity)
    if i >= 4:
        return 'high'
    if i >= 2:
        return 'medium'
    return 'low'


def bucket_stress(stress: float) -> str:
    """Convert stress level (1-5) to bucket."""
    s = float(stress)
    if s >= 4:
        return 'high'
    if s >= 2:
        return 'medium'
    return 'low'


def normalize_state(state: str) -> str:
    """Map raw emotional state string to a rule-engine group."""
    s = state.lower().strip()
    mapping = {
        'anxious': 'anxious', 'anxiety': 'anxious', 'nervous': 'anxious',
        'worried': 'anxious', 'uneasy': 'anxious',
        'stressed': 'stressed', 'overwhelmed': 'stressed', 'burnt_out': 'stressed',
        'sad': 'sad', 'melancholy': 'sad', 'depressed': 'sad', 'low': 'sad',
        'gloomy': 'sad', 'down': 'sad',
        'calm': 'calm', 'peaceful': 'calm', 'relaxed': 'calm',
        'serene': 'calm', 'tranquil': 'calm',
        'happy': 'happy', 'joyful': 'happy', 'elated': 'happy', 'cheerful': 'happy',
        'excited': 'excited', 'energetic': 'excited', 'enthusiastic': 'excited',
        'content': 'content', 'satisfied': 'content', 'grateful': 'content',
        'tired': 'tired', 'exhausted': 'tired', 'sleepy': 'tired', 'drained': 'tired',
        'fatigued': 'fatigued',
        'confused': 'confused', 'uncertain': 'confused', 'lost': 'confused',
        'restless': 'restless', 'agitated': 'restless', 'fidgety': 'restless',
        'neutral': 'default', 'mixed': 'default', 'focused': 'calm',
    }
    return mapping.get(s, 'default')


def normalize_time(time_str: str) -> str:
    """Map time of day string to a rule-engine group."""
    t = str(time_str).lower().strip()
    if any(x in t for x in ['morning', 'dawn', 'am', 'early']):
        return 'morning'
    if any(x in t for x in ['afternoon', 'midday', 'noon']):
        return 'afternoon'
    if any(x in t for x in ['evening', 'dusk', 'pm']):
        return 'evening'
    if any(x in t for x in ['night', 'midnight', 'late']):
        return 'night'
    return 'default'


def decide_what(state_str: str, intensity: float, stress: float) -> str:
    """
    Decide what action the user should take.

    Args:
        state_str: Predicted emotional state
        intensity: Predicted intensity (1-5)
        stress: Stress level (1-5)

    Returns:
        Recommended action (e.g., 'box_breathing', 'deep_work')
    """
    sg = normalize_state(state_str)
    ib = bucket_intensity(intensity)
    sb = bucket_stress(stress)

    # Try increasingly general keys
    for key in [
        (sg, ib, sb),
        (sg, ib, 'any'),
        (sg, 'any', sb),
        (sg, 'any', 'any'),
        ('default', 'any', 'any'),
    ]:
        if key in WHAT_RULES:
            return WHAT_RULES[key]
    return 'pause'


def decide_when(time_str: str, intensity: float, what_action: str) -> str:
    """
    Decide when the user should perform the action.

    Args:
        time_str: Time of day
        intensity: Predicted intensity (1-5)
        what_action: The recommended action

    Returns:
        Timing recommendation (e.g., 'now', 'within_15_min')
    """
    tg = normalize_time(time_str)
    ib = bucket_intensity(intensity)
    at = ACTION_TYPE_MAP.get(what_action, 'any')

    # Try increasingly general keys
    for key in [
        (tg, ib, at),
        (tg, ib, 'any'),
        (tg, 'any', at),
        (tg, 'any', 'any'),
        ('default', 'any', 'any'),
    ]:
        if key in WHEN_RULES:
            return WHEN_RULES[key]
    return 'within_15_min'


def get_decision(state: str, intensity: float, stress: float, time_of_day: str) -> Tuple[str, str]:
    """
    Get both what to do and when to do it.

    Args:
        state: Predicted emotional state
        intensity: Predicted intensity (1-5)
        stress: Stress level (1-5)
        time_of_day: Time of day

    Returns:
        Tuple of (what_to_do, when_to_do)
    """
    what = decide_what(state, intensity, stress)
    when = decide_when(time_of_day, intensity, what)
    return what, when

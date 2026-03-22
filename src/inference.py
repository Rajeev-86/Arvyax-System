"""Main inference pipeline for emotional state and decision guidance."""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Optional, Tuple

from .preprocessing import preprocess, preprocess_single
from .feature_engineering import FeatureEngineer, create_feature_engineer
from .decision_engine import get_decision
from .uncertainty import compute_confidence, compute_confidence_single
from .config import CLF_STATE_PATH, REG_INTENSITY_PATH, STATE_ENCODER_PATH, DEFAULT_SLEEP_MEDIAN
from .message_generator import MessageGenerator


class EmotionalInferencePipeline:
    """End-to-end inference pipeline for emotional intelligence system."""

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        use_slm: bool = False,
        use_template_fallback: bool = True
    ):
        """
        Initialize the inference pipeline.

        Args:
            models_dir: Directory containing saved models
            use_slm: Whether to use SLM for message generation
            use_template_fallback: Whether to fallback to templates if SLM fails
        """
        # Load models
        self.clf_state = joblib.load(CLF_STATE_PATH)
        self.reg_intensity = joblib.load(REG_INTENSITY_PATH)
        self.state_encoder = joblib.load(STATE_ENCODER_PATH)

        # Initialize feature engineer
        self.feature_engineer = create_feature_engineer()

        # Initialize message generator
        self.message_gen = MessageGenerator(
            use_slm=use_slm,
            use_template_fallback=use_template_fallback
        )

        print("✅ EmotionalInferencePipeline initialized")

    def predict_single(
        self,
        data: Dict,
        generate_message: bool = False
    ) -> Dict:
        """
        Make prediction for a single input record.

        Args:
            data: Dict with required fields:
                - journal_text (str)
                - ambience_type (str)
                - duration_min (int)
                - sleep_hours (float, optional)
                - energy_level (int)
                - stress_level (int)
                - time_of_day (str)
                - previous_day_mood (str, optional)
                - face_emotion_hint (str, optional)
                - reflection_quality (str)
            generate_message: Whether to generate supportive message

        Returns:
            Dict with predictions:
                - predicted_state (str)
                - predicted_intensity (int)
                - confidence (float)
                - uncertain_flag (int)
                - what_to_do (str)
                - when_to_do (str)
                - message (str, optional)
        """
        # Preprocess single record
        processed = preprocess_single(data, sleep_median=DEFAULT_SLEEP_MEDIAN)

        # Encode categorical features
        processed = self.feature_engineer.encode_categorical_single(processed)

        # Get text embedding
        embedding = self.feature_engineer.encode_text_single(processed['journal_text'])

        # Build feature vector
        X = self.feature_engineer.build_feature_vector(processed, embedding)

        # Predict emotional state
        state_proba = self.clf_state.predict_proba(X.reshape(1, -1))[0]
        state_label = np.argmax(state_proba)
        state_str = self.state_encoder.inverse_transform([state_label])[0]

        # Compute confidence
        confidence, uncertain_flag = compute_confidence_single(state_proba)

        # Predict intensity
        intensity_raw = self.reg_intensity.predict(X.reshape(1, -1))[0]
        intensity = int(np.clip(np.round(intensity_raw), 1, 5))

        # Get decision
        stress = float(processed.get('stress_level', 3))
        time_of_day = str(processed.get('time_of_day', 'afternoon'))
        what_to_do, when_to_do = get_decision(state_str, intensity, stress, time_of_day)

        # Generate message if requested
        message = None
        if generate_message:
            message = self.message_gen.generate(
                state=state_str,
                intensity=intensity,
                action=what_to_do,
                confidence=confidence,
                uncertain=uncertain_flag
            )

        return {
            'predicted_state': state_str,
            'predicted_intensity': intensity,
            'confidence': float(confidence),
            'uncertain_flag': int(uncertain_flag),
            'what_to_do': what_to_do,
            'when_to_do': when_to_do,
            'message': message,
            'confidence_score': confidence,
            'state_probabilities': {
                self.state_encoder.classes_[i]: float(state_proba[i])
                for i in range(len(self.state_encoder.classes_))
            }
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        generate_messages: bool = False
    ) -> pd.DataFrame:
        """
        Make predictions for a batch of records.

        Args:
            df: DataFrame with required columns
            generate_messages: Whether to generate messages for each prediction

        Returns:
            DataFrame with predictions
        """
        # Preprocess all records
        df_processed = preprocess(df, sleep_median=DEFAULT_SLEEP_MEDIAN)

        # Encode categorical features
        df_processed = self.feature_engineer.encode_categorical(df_processed)

        # Get embeddings
        embeddings = self.feature_engineer.encode_text(df_processed['journal_text'].tolist())

        # Build feature matrix
        X = self.feature_engineer.build_feature_matrix(df_processed, embeddings)

        # Predict states
        state_probas = self.clf_state.predict_proba(X)
        state_labels = np.argmax(state_probas, axis=1)
        state_strings = self.state_encoder.inverse_transform(state_labels)

        # Compute confidence
        confidences, uncertain_flags = compute_confidence(state_probas)

        # Predict intensities
        intensity_raw = self.reg_intensity.predict(X)
        intensities = np.clip(np.round(intensity_raw), 1, 5).astype(int)

        # Get decisions
        what_to_dos = []
        when_to_dos = []
        for i in range(len(df_processed)):
            what, when = get_decision(
                state_strings[i],
                intensities[i],
                float(df_processed['stress_level'].iloc[i]),
                str(df_processed['time_of_day'].iloc[i])
            )
            what_to_dos.append(what)
            when_to_dos.append(when)

        # Generate messages if requested
        messages = []
        if generate_messages:
            for i in range(len(df_processed)):
                msg = self.message_gen.generate(
                    state=state_strings[i],
                    intensity=intensities[i],
                    action=what_to_dos[i],
                    confidence=confidences[i],
                    uncertain=uncertain_flags[i]
                )
                messages.append(msg)

        # Assemble results
        results = pd.DataFrame({
            'predicted_state': state_strings,
            'predicted_intensity': intensities,
            'confidence': confidences,
            'uncertain_flag': uncertain_flags,
            'what_to_do': what_to_dos,
            'when_to_do': when_to_dos,
        })

        if generate_messages:
            results['message'] = messages

        # Preserve original IDs if present
        if 'id' in df.columns:
            results.insert(0, 'id', df['id'].values)

        return results

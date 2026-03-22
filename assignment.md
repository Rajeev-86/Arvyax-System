Training Data:  https://docs.google.com/spreadsheets/d/1yLDum7yWr3IH0KivluCBEvqHGlfvFW_S/edit?usp=sharing&ouid=107676686611527271344&rtpof=true&sd=true

Test Data: https://docs.google.com/spreadsheets/d/1lCvTufEhGgtDJp6b9oYyFXpCZqWPirSX/edit?usp=sharing&ouid=107676686611527271344&rtpof=true&sd=true

🌿 ArvyaX Machine Learning Internship Assignment
Team ArvyaX · RevoltronX
Theme: From Understanding Humans → To Guiding Them

At ArvyaX, we are building AI systems that go beyond prediction.

We aim to create intelligence that can:

understand human emotional state

reason under imperfect and noisy signals

decide meaningful next actions

guide users toward better mental states


After immersive sessions (forest, ocean, rain, mountain, café), users write short reflections.

These reflections are:

messy

short or vague

sometimes contradictory


We also collect lightweight contextual signals:

sleep

stress

energy

time of day

previous mood


⚠️ Important : This is NOT a standard classification problem.

Real-world systems must handle:

noisy text

missing data

conflicting signals

imperfect labels

Your goal is to build a system that can understand → decide → guide.

 Objective

Build a system that takes user input and produces:

1. Emotional Understanding
predicted_state

predicted_intensity (1–5)


2. Decision Layer (Core)
Your system must decide:

➤ What should the user do?

(e.g., breathing, journaling, deep work, rest)

➤ When should they do it?

now

within_15_min

later_today

tonight

tomorrow_morning


3. Uncertainty Awareness
For each prediction:

confidence (0–1)

uncertain_flag (0 or 1)

👉 A strong system knows when it is unsure.


4. (Optional Bonus) Supportive Message
Generate a short human-like response explaining the recommendation.

Example:

“You seem slightly restless right now. Let’s slow things down. Try a short breathing exercise before planning your day.”

Dataset
You are provided with a dataset containing: id, journal_text, ambience_type, duration_min, sleep_hours, energy_level, stress_level, time_of_day, previous_day_mood, face_emotion_hint, reflection_quality, emotional_state, intensity

Tasks

Part 1 — Emotional State Prediction
Predict:

emotional_state

Part 2 — Intensity Prediction
Predict:

intensity

Explain whether you treat this as:

classification

regression

Part 3 — Decision Engine (What + When)
Design logic to decide:

What to do (to make him ready for user intention deep work, sleep etc)
Examples: box_breathing, ournaling. grounding, deep_work, yoga. sound therapy, light_planning, rest, movement, pause

When to do it
Options: now, within_15_min, later_today, tonight, tomorrow_morning

Your system should use: predicted state, intensity, stress, energy, time of day

Part 4 — Uncertainty Modeling
Provide:

confidence score

uncertain flag

Part 5 — Feature Understanding
Explain:

what features mattered most

text vs metadata importance

Part 6 — Ablation Study
Compare:

text-only model

text + metadata model


Part 7 — Error Analysis (Very Important)
Analyze at least 10 failure cases.

Explain:

what went wrong

why the model failed

how to improve

Focus on:

ambiguous text

conflicting signals

short inputs

noisy labels


Part 8 — Edge / Offline Thinking
Explain how your system would run:

on mobile

on-device

Discuss:

model size

latency

tradeoffs

Part 9 — Robustness (Small but Important)
Explain how your system handles:

very short text (“ok”, “fine”)

missing values

contradictory inputs

Constraints
Allowed
scikit-learn

XGBoost

PyTorch

TensorFlow

local lightweight models

Not Allowed
OpenAI / Gemini / Claude APIs

any hosted LLM

* Your solution must run locally.

Deliverables
1. Code
End-to-end pipeline

2. predictions.csv
id

predicted_state

predicted_intensity

confidence

uncertain_flag

what_to_do

when_to_do

3. README.md
Include:

setup instructions

approach

feature engineering

model choice

how to run

4. ERROR_ANALYSIS.md
Include:

10 failure cases

insights

5. EDGE_PLAN.md
Explain:

deployment approach

optimizations

Evaluation Criteria
ML + reasoning - 20%

Decision logic (what + when) - 20%

Uncertainty handling- 15%

Error analysis depth - 15%

Feature understanding- 10%

Code quality- 10%

Edge thinking- 10%

What We Care About
We are looking for:

real-world thinking

ability to handle messy data

reasoning under uncertainty

meaningful decision-making

product-oriented mindset

Not just accuracy.

 Bonus (Optional)
supportive conversational message

small local API (Flask/FastAPI)

simple UI demo

lightweight conversational model (SLM)

label noise handling


⚠️ Final Note
During the interview, you will be asked to:

explain your model

justify decisions

walk through failure cases


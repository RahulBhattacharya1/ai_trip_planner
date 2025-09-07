# app.py — Streamlit AI Trip Planner (UI-only)
# Features: Provider switch (OpenAI / Offline), model picker, brand color, temperature, max tokens
# Output sections: Top Attractions, Top Hiking Trails
# Notes:
# - For OpenAI, set OPENAI_API_KEY in your environment (or via sidebar input if you prefer).
# - Offline mode is deterministic and needs no network.

import os
import json
import random
from typing import List, Tuple
import streamlit as st

# ============ Page ============

st.set_page_config(page_title="AI Trip Planner", layout="wide")

# Simple brand color helper
def brand_h2(text: str, color: str):
    st.markdown(f"<h2 style='margin:0 0 .5rem 0; color:{color}'>{text}</h2>", unsafe_allow_html=True)

def brand_h3(text: str, color: str):
    st.markdown(f"<h3 style='margin-top:1.25rem; margin-bottom:.25rem; color:{color}'>{text}</h3>", unsafe_allow_html=True)

st.title("AI Trip Planner")

# ============ Sidebar ============

with st.sidebar:
    st.subheader("Generator")
    provider = st.selectbox("Provider", ["OpenAI", "Offline (rule-based)"])
    model = st.selectbox("Model (OpenAI)", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
    brand = st.color_picker("Brand color", value="#0F62FE")
    temp = st.slider("Creativity (OpenAI)", 0.0, 1.0, 0.4, 0.05)
    max_tokens = st.slider("Max tokens (OpenAI)", 256, 4096, 1200, 32)
    st.caption("Tip: Use Offline to demo your engineering logic without any API.")
    # Optional API key field (fallback to env)
    show_key = st.checkbox("Enter OpenAI API key here")
    api_key_local = st.text_input("OPENAI_API_KEY", type="password") if show_key else ""

# ============ Inputs ============

colA, colB = st.columns([1.3, 1])
with colA:
    destination = st.text_input(
        "Destination (city, region, or park)",
        placeholder="e.g., Gatlinburg, TN or Rocky Mountain National Park"
    )
    days = st.slider("Number of days", 1, 14, 5)
    season = st.selectbox("Season", ["Any", "Spring", "Summer", "Fall", "Winter"])
    travel_style = st.select_slider("Pace", options=["Relaxed", "Balanced", "Packed"], value="Balanced")
    budget = st.selectbox("Budget", ["Any", "$", "$$", "$$$"])
with colB:
    interests = st.multiselect(
        "Interests (optional)",
        [
            "Nature", "Scenic Drives", "Waterfalls", "Wildlife",
            "Museums", "Local Food", "Photography", "Historic Sites",
            "Lakes", "Caves", "Coastal Views", "Sunrise/Sunset"
        ],
        default=["Nature", "Scenic Drives", "Photography"]
    )
    difficulty = st.selectbox("Hiking difficulty preference", ["Any", "Easy", "Moderate", "Hard"])
    need_kid_friendly = st.checkbox("Kid-friendly options")
    need_elder_friendly = st.checkbox("Elder-friendly options")
    need_wheelchair = st.checkbox("Wheelchair accessible options")

col1, col2, col3 = st.columns([1,1,1])
with col1:
    gen = st.button("Generate Plan", type="primary")
with col2:
    regen = st.button("Regenerate Suggestions")
with col3:
    clear = st.button("Clear")

if "seed" not in st.session_state:
    st.session_state.seed = 42

if clear:
    st.session_state.pop("results", None)

# ============ Offline Generator (Deterministic) ============

OFFLINE_ATTRACTIONS = [
    "Scenic Overlook", "Historic Downtown", "Local Artisan Market", "Regional History Museum",
    "Riverwalk Promenade", "Botanical Garden", "Wildlife Viewing Area", "Cultural Heritage Center",
    "Iconic Bridge/Walkway", "Lakeside Boardwalk", "Panoramic Viewpoint", "Visitor Center Exhibits"
]

OFFLINE_HIKES = [
    ("Nature Loop", "Easy", "1.5 mi", "Loop"),
    ("Waterfall Trail", "Moderate", "3.2 mi", "Out & Back"),
    ("Summit Ridge", "Hard", "6.0 mi", "Out & Back"),
    ("Canyon Path", "Moderate", "4.1 mi", "Loop"),
    ("Lakeshore Walk", "Easy", "2.3 mi", "Loop"),
    ("Wildflower Route", "Easy", "1.8 mi", "Out & Back"),
    ("Overlook Climb", "Hard", "5.5 mi", "Loop")
]

def offline_trip_plan(
    destination: str,
    days: int,
    difficulty: str,
    seed: int
) -> Tuple[List[str], List[dict]]:
    rng = random.Random(seed + len(destination) + days)
    # Attractions
    base = [f"{destination} {name}" for name in OFFLINE_ATTRACTIONS]
    rng.shuffle(base)
    top_attractions = base[: min(7, max(3, days + 1))]

    # Hikes
    hikes = []
    for name, diff, dist, kind in OFFLINE_HIKES:
        # Filter by difficulty preference (if set)
        if difficulty != "Any" and diff != difficulty:
            continue
        hikes.append(
            {
                "name": f"{destination} {name}",
                "difficulty": diff,
                "distance": dist,
                "type": kind
            }
        )
    if not hikes:
        # If filter too strict, relax to any
        for name, diff, dist, kind in OFFLINE_HIKES:
            hikes.append(
                {
                    "name": f"{destination} {name}",
                    "difficulty": diff,
                    "distance": dist,
                    "type": kind
                }
            )
    rng.shuffle(hikes)
    top_hikes = hikes[: min(7, max(3, days + 1))]
    return top_attractions, top_hikes

# ============ OpenAI Generator ============

def call_openai(
    model: str,
    api_key: str,
    destination: str,
    days: int,
    season: str,
    travel_style: str,
    budget: str,
    interests: List[str],
    difficulty: str,
    kid: bool,
    elder: bool,
    wheelchair: bool,
    temp: float,
    max_tokens: int
) -> Tuple[List[str], List[dict]]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    constraints = []
    if kid: constraints.append("kid-friendly")
    if elder: constraints.append("elder-friendly")
    if wheelchair: constraints.append("wheelchair accessible")

    prompt = {
        "role": "user",
        "content": (
            "You are a precise trip-planning assistant. "
            "Return strict JSON with two keys: attractions (list of strings) and hikes (list of objects with keys: name, difficulty, distance, type). "
            "Do not include commentary or extra keys.\n\n"
            f"Destination: {destination}\n"
            f"Days: {days}\n"
            f"Season: {season}\n"
            f"Pace: {travel_style}\n"
            f"Budget: {budget}\n"
            f"Interests: {', '.join(interests) if interests else 'Any'}\n"
            f"Hiking difficulty preference: {difficulty}\n"
            f"Accessibility constraints: {', '.join(constraints) if constraints else 'None'}\n"
            "Aim for 5–10 attractions and 5–10 hikes when possible.\n"
            "Keep attraction names concise and recognizable; hikes must include difficulty, distance (with units), and route type."
        )
    }

    resp = client.chat.completions.create(
        model=model,
        temperature=float(temp),
        max_tokens=int(max_tokens),
        messages=[
            {"role": "system", "content": "You output minimal JSON only."},
            prompt
        ]
    )
    text = resp.choices[0].message.content.strip()

    # Attempt to parse JSON; if model wrapped it in markdown, strip fences
    if text.startswith("```"):
        text = text.strip("`")
        # after stripping, content may look like json\n{...}
        if "\n" in text:
            text = text.split("\n", 1)[1].strip()
    data = json.loads(text)

    # Validate shapes
    attractions = [str(x) for x in data.get("attractions", [])][:10]
    hikes_raw = data.get("hikes", [])
    hikes = []
    for h in hikes_raw:
        if not isinstance(h, dict):
            continue
        hikes.append(
            {
                "name": str(h.get("name", "")),
                "difficulty": str(h.get("difficulty", "")),
                "distance": str(h.get("distance", "")),
                "type": str(h.get("type", "")),
            }
        )
    hikes = hikes[:10]
    return attractions, hikes

# ============ Orchestrator ============

def generate(
    provider: str,
    model: str,
    destination: str,
    days: int,
    season: str,
    travel_style: str,
    budget: str,
    interests: List[str],
    difficulty: str,
    kid: bool,
    elder: bool,
    wheelchair: bool,
    temp: float,
    max_tokens: int
) -> Tuple[List[str], List[dict], str]:
    if provider == "Offline (rule-based)":
        attractions, hikes = offline_trip_plan(destination, days, difficulty, st.session_state.seed)
        return attractions, hikes, "offline"
    else:
        api_key = (api_key_local or os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            st.warning("OPENAI_API_KEY is missing. Falling back to Offline mode.")
            attractions, hikes = offline_trip_plan(destination, days, difficulty, st.session_state.seed)
            return attractions, hikes, "offline-fallback"
        try:
            attractions, hikes = call_openai(
                model=model,
                api_key=api_key,
                destination=destination,
                days=days,
                season=season,
                travel_style=travel_style,
                budget=budget,
                interests=interests,
                difficulty=difficulty,
                kid=kid,
                elder=elder,
                wheelchair=wheelchair,
                temp=temp,
                max_tokens=max_tokens
            )
            if not attractions and not hikes:
                raise ValueError("Empty result")
            return attractions, hikes, "openai"
        except Exception as e:
            st.error(f"OpenAI error: {e}. Using Offline mode.")
            attractions, hikes = offline_trip_plan(destination, days, difficulty, st.session_state.seed)
            return attractions, hikes, "offline-fallback"

# ============ UI Actions ============

def render_results(attractions: List[str], hikes: List[dict], brand: str):
    brand_h2("Top Attractions", brand)
    if not attractions:
        st.info("No attractions found.")
    else:
        cols = st.columns(2)
        for i, a in enumerate(attractions):
            with cols[i % 2]:
                st.markdown(
                    f"""
<div style="border:1px solid #e5e7eb; padding:.75rem 1rem; border-radius:10px; margin-bottom:.75rem;">
  <div style="font-weight:600">{a}</div>
  <div style="font-size:.9rem; opacity:.8;">
    <a href="https://www.google.com/maps/search/{a.replace(' ', '+')}" target="_blank">View on Maps</a> ·
    <a href="https://www.google.com/search?q={a.replace(' ', '+')}" target="_blank">Web Search</a>
  </div>
</div>
                    """,
                    unsafe_allow_html=True
                )

    brand_h2("Top Hiking Trails", brand)
    if not hikes:
        st.info("No hiking trails found.")
    else:
        cols = st.columns(2)
        for i, h in enumerate(hikes):
            name = h.get("name", "").strip()
            diff = h.get("difficulty", "").strip()
            dist = h.get("distance", "").strip()
            rtype = h.get("type", "").strip()
            with cols[i % 2]:
                st.markdown(
                    f"""
<div style="border:1px solid #e5e7eb; padding:.75rem 1rem; border-radius:10px; margin-bottom:.75rem;">
  <div style="font-weight:600">{name}</div>
  <div style="font-size:.95rem; margin:.2rem 0;">Difficulty: {diff} · Distance: {dist} · Route: {rtype}</div>
  <div style="font-size:.9rem; opacity:.8;">
    <a href="https://www.google.com/maps/search/{name.replace(' ', '+')}" target="_blank">Trailhead Maps</a> ·
    <a href="https://www.google.com/search?q={name.replace(' ', '+')}" target="_blank">Trail Info</a>
  </div>
</div>
                    """,
                    unsafe_allow_html=True
                )

    with st.expander("Copy as Markdown"):
        md = []
        md.append("## Top Attractions")
        for a in attractions:
            md.append(f"- {a}")
        md.append("\n## Top Hiking Trails")
        for h in hikes:
            md.append(f"- {h.get('name','')} — {h.get('difficulty','')} · {h.get('distance','')} · {h.get('type','')}")
        st.code("\n".join(md), language="markdown")

# ============ Main Trigger ============

if (gen or regen) and destination.strip():
    results = generate(
        provider=provider,
        model=model,
        destination=destination.strip(),
        days=days,
        season=season,
        travel_style=travel_style,
        budget=budget,
        interests=interests,
        difficulty=difficulty,
        kid=need_kid_friendly,
        elder=need_elder_friendly,
        wheelchair=need_wheelchair,
        temp=temp,
        max_tokens=max_tokens
    )
    st.session_state.results = results
    # nudge the seed for regeneration variety in offline
    if regen:
        st.session_state.seed += 7

if "results" in st.session_state:
    attractions, hikes, mode = st.session_state.results
    st.caption(f"Mode: {mode}")
    # Section header with destination and meta
    brand_h2(f"Trip Plan: {destination} · {days} days", brand)
    st.write(
        f"Pace: {travel_style} | Budget: {budget} | Season: {season} | "
        f"Interests: {', '.join(interests) if interests else 'Any'} | "
        f"Hike difficulty: {difficulty}"
    )
    render_results(attractions, hikes, brand)
else:
    st.info("Enter a destination and click Generate Plan.")

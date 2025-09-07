# app.py — Streamlit AI Trip Planner (UI-only, OpenAI via st.secrets)
# Mirrors your sample's patterns: st.secrets["OPENAI_API_KEY"], rate limits, sidebar add-ons.

import os
import time
import json
import random
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple

import streamlit as st

# Optional: only needed if provider == "OpenAI"
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ======================= App Config =======================
st.set_page_config(page_title="AI Trip Planner", layout="wide")

# ======================= Rate Limiting (mirrors your sample) =======================
# === Runtime budget/limits loader (auto-updates from GitHub) ===

import os, time, types, urllib.request
import streamlit as st

# Raw URL of your budget.py in the shared repo (override via env if needed)
BUDGET_URL = os.getenv(
    "BUDGET_URL",
    "https://raw.githubusercontent.com/RahulBhattacharya1/shared_config/main/budget.py",
)

# Safe defaults if the fetch fails
_BUDGET_DEFAULTS = {
    "COOLDOWN_SECONDS": 30,
    "DAILY_LIMIT": 40,
    "HOURLY_SHARED_CAP": 250,
    "DAILY_BUDGET": 1.00,
    "EST_COST_PER_GEN": 1.00,
    "VERSION": "fallback-local",
}

def _fetch_remote_budget(url: str) -> dict:
    mod = types.ModuleType("budget_remote")
    with urllib.request.urlopen(url, timeout=5) as r:
        code = r.read().decode("utf-8")
    exec(compile(code, "budget_remote", "exec"), mod.__dict__)
    cfg = {}
    for k in _BUDGET_DEFAULTS.keys():
        cfg[k] = getattr(mod, k, _BUDGET_DEFAULTS[k])
    return cfg

def get_budget(ttl_seconds: int = 300) -> dict:
    """Fetch and cache remote budget in session state with a TTL."""
    now = time.time()
    cache = st.session_state.get("_budget_cache")
    ts = st.session_state.get("_budget_cache_ts", 0)

    if cache and (now - ts) < ttl_seconds:
        return cache

    try:
        cfg = _fetch_remote_budget(BUDGET_URL)
    except Exception:
        cfg = _BUDGET_DEFAULTS.copy()

    # Allow env overrides if you want per-deploy tuning
    cfg["DAILY_BUDGET"] = float(os.getenv("DAILY_BUDGET", cfg["DAILY_BUDGET"]))
    cfg["EST_COST_PER_GEN"] = float(os.getenv("EST_COST_PER_GEN", cfg["EST_COST_PER_GEN"]))

    st.session_state["_budget_cache"] = cfg
    st.session_state["_budget_cache_ts"] = now
    return cfg

# Load once (respects TTL); you can expose a "Refresh config" button to clear cache
_cfg = get_budget(ttl_seconds=300)

COOLDOWN_SECONDS  = int(_cfg["COOLDOWN_SECONDS"])
DAILY_LIMIT       = int(_cfg["DAILY_LIMIT"])
HOURLY_SHARED_CAP = int(_cfg["HOURLY_SHARED_CAP"])
DAILY_BUDGET      = float(_cfg["DAILY_BUDGET"])
EST_COST_PER_GEN  = float(_cfg["EST_COST_PER_GEN"])
CONFIG_VERSION    = str(_cfg.get("VERSION", "unknown"))
# === End runtime loader ===

def _hour_bucket(now=None):
    now = now or dt.datetime.utcnow()
    return now.strftime("%Y-%m-%d-%H")

@st.cache_resource
def _shared_hourly_counters():
    # In-memory dict shared by all sessions in this Streamlit process
    # key: "YYYY-MM-DD-HH", value: int count
    return {}

def init_rate_limit_state():
    ss = st.session_state
    today = dt.date.today().isoformat()
    if "rl_date" not in ss or ss["rl_date"] != today:
        ss["rl_date"] = today
        ss["rl_calls_today"] = 0
        ss["rl_last_ts"] = 0.0
    if "rl_last_ts" not in ss:
        ss["rl_last_ts"] = 0.0
    if "rl_calls_today" not in ss:
        ss["rl_calls_today"] = 0

def can_call_now():
    init_rate_limit_state()
    ss = st.session_state
    now = time.time()

    # Cooldown guard
    remaining = int(max(0, ss["rl_last_ts"] + COOLDOWN_SECONDS - now))
    if remaining > 0:
        return (False, f"Please wait {remaining}s before the next generation.", remaining)

    # === NEW: Daily budget guardrail ===
    # Uses shared values loaded at runtime: DAILY_BUDGET and EST_COST_PER_GEN
    est_spend = ss["rl_calls_today"] * EST_COST_PER_GEN
    if est_spend >= DAILY_BUDGET:
        return (False, f"Daily cost limit reached (${DAILY_BUDGET:.2f}). Try again tomorrow.", 0)

    # Per-session daily cap (still keeps your old guard)
    if ss["rl_calls_today"] >= DAILY_LIMIT:
        return (False, f"Daily limit reached ({DAILY_LIMIT} generations). Try again tomorrow.", 0)

    # Optional shared hourly cap
    if HOURLY_SHARED_CAP > 0:
        bucket = _hour_bucket()
        counters = _shared_hourly_counters()
        used = counters.get(bucket, 0)
        if used >= HOURLY_SHARED_CAP:
            return (False, "Hourly capacity reached. Please try later.", 0)

    return (True, "", 0)
    
def record_successful_call():
    ss = st.session_state
    ss["rl_last_ts"] = time.time()
    ss["rl_calls_today"] += 1

    if HOURLY_SHARED_CAP > 0:
        bucket = _hour_bucket()
        counters = _shared_hourly_counters()
        counters[bucket] = counters.get(bucket, 0) + 1

# ======================= Data Models =======================
@dataclass
class Hike:
    name: str
    difficulty: str
    distance: str
    type: str

# ======================= UI Helpers =======================
def brand_h2(text: str, color: str):
    st.markdown(f"<h2 style='margin:.25rem 0 .75rem 0; color:{color}'>{text}</h2>", unsafe_allow_html=True)

def section_card(title: str, subtitle_html: str = "", links: List[Tuple[str, str]] = None):
    links = links or []
    items = " · ".join(f'<a href="{href}" target="_blank">{label}</a>' for label, href in links)
    st.markdown(
        f"""
<div style="border:1px solid #e5e7eb; padding:.75rem 1rem; border-radius:10px; margin-bottom:.75rem;">
  <div style="font-weight:600">{title}</div>
  {f'<div style="font-size:.95rem; margin:.2rem 0;">{subtitle_html}</div>' if subtitle_html else ''}
  {f'<div style="font-size:.9rem; opacity:.85;">{items}</div>' if items else ''}
</div>
        """,
        unsafe_allow_html=True
    )

# ======================= Offline Generator =======================
OFFLINE_ATTRACTIONS = [
    "Scenic Overlook", "Historic Downtown", "Local Artisan Market", "Regional History Museum",
    "Riverwalk Promenade", "Botanical Garden", "Wildlife Viewing Area", "Cultural Heritage Center",
    "Iconic Bridge", "Lakeside Boardwalk", "Panoramic Viewpoint", "Visitor Center Exhibits"
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

def offline_trip_plan(destination: str, days: int, difficulty: str, seed: int):
    rng = random.Random(seed + len(destination) + days)
    # Attractions
    base = [f"{destination} {name}" for name in OFFLINE_ATTRACTIONS]
    rng.shuffle(base)
    top_attractions = base[: min(7, max(3, days + 1))]

    # Hikes with difficulty filtering
    hikes = []
    for name, diff, dist, rtype in OFFLINE_HIKES:
        if difficulty != "Any" and diff != difficulty:
            continue
        hikes.append(
            Hike(
                name=f"{destination} {name}",
                difficulty=diff,
                distance=dist,
                type=rtype
            )
        )
    if not hikes:
        for name, diff, dist, rtype in OFFLINE_HIKES:
            hikes.append(Hike(name=f"{destination} {name}", difficulty=diff, distance=dist, type=rtype))
    rng.shuffle(hikes)
    top_hikes = hikes[: min(7, max(3, days + 1))]
    return top_attractions, top_hikes

# ======================= OpenAI Call (st.secrets) =======================
def call_openai(
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
    temperature: float,
    max_tokens: int
):
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in Streamlit Secrets.")
    if OpenAI is None:
        raise RuntimeError("openai package not available. Add openai to requirements.txt.")

    client = OpenAI(api_key=api_key)

    constraints = []
    if kid: constraints.append("kid-friendly")
    if elder: constraints.append("elder-friendly")
    if wheelchair: constraints.append("wheelchair accessible")

    sys = (
        "You are a precise trip-planning assistant.\n"
        "Return JSON only with two keys: 'attractions' (list of strings) and 'hikes' "
        "(list of objects with keys: name, difficulty, distance, type).\n"
        "No extra keys, no prose, no markdown fences."
    )
    usr = (
        f"Destination: {destination}\n"
        f"Days: {days}\n"
        f"Season: {season}\n"
        f"Pace: {travel_style}\n"
        f"Budget: {budget}\n"
        f"Interests: {', '.join(interests) if interests else 'Any'}\n"
        f"Hiking difficulty preference: {difficulty}\n"
        f"Accessibility constraints: {', '.join(constraints) if constraints else 'None'}\n"
        "Aim for 5–10 attractions and 5–10 hikes when possible. "
        "Each hike must include difficulty, distance with units, and route type."
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": usr}
        ]
    )
    text = resp.choices[0].message.content.strip()

    # Strip accidental code fences
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1].strip()

    data = json.loads(text)
    attractions = [str(x) for x in data.get("attractions", [])][:10]
    hikes_raw = data.get("hikes", [])
    hikes: List[Hike] = []
    for h in hikes_raw:
        if isinstance(h, dict):
            hikes.append(
                Hike(
                    name=str(h.get("name", "")),
                    difficulty=str(h.get("difficulty", "")),
                    distance=str(h.get("distance", "")),
                    type=str(h.get("type", ""))
                )
            )
    hikes = hikes[:10]
    return attractions, hikes

# ======================= Inputs & Sidebar =======================
st.title("AI Trip Planner")

with st.sidebar:
    st.subheader("Generator")
    provider = st.selectbox("Provider", ["OpenAI", "Offline (rule-based)"])
    model = st.selectbox("Model (OpenAI)", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"])
    brand = "#0F62FE"
    temp = st.slider("Creativity (OpenAI)", 0.0, 1.0, 0.4, 0.05)
    max_tokens = st.slider("Max tokens (OpenAI)", 256, 4096, 1200, 32)

    # Usage panel (rate limits)
    init_rate_limit_state()
    ss = st.session_state
    st.markdown("**Usage limits**")
    st.write(f"<span style='font-size:0.9rem'>Today: {ss['rl_calls_today']} / {DAILY_LIMIT} generations</span>")
    if HOURLY_SHARED_CAP > 0:
        counters = _shared_hourly_counters()
        used = counters.get(_hour_bucket(), 0)
        st.write(f"<span style='font-size:0.9rem'>Hour capacity: {used} / {HOURLY_SHARED_CAP}</span>")
    remaining = int(max(0, ss["rl_last_ts"] + COOLDOWN_SECONDS - time.time()))
    if remaining > 0:
        st.progress(min(1.0, (COOLDOWN_SECONDS - remaining) / COOLDOWN_SECONDS))
        st.caption(f"Cooldown: {remaining}s")
    est_spend = ss['rl_calls_today'] * EST_COST_PER_GEN
    st.markdown(
        f"<span style='font-size:0.9rem'>Budget: &#36;{est_spend:.2f} / &#36;{DAILY_BUDGET:.2f}</span><br/>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<span style='font-size:0.8rem; opacity:0.8'>Version: {CONFIG_VERSION}</span>",
        unsafe_allow_html=True
    )
    
    # Optional: show a warning if we’re on fallback defaults (remote fetch failed)
    if CONFIG_VERSION == "fallback-local":
        st.warning("Using fallback defaults — couldn’t fetch remote budget.py")

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

col1, col2, col3 = st.columns([1, 1, 1])
allowed, reason, _wait = can_call_now()
with col1:
    gen = st.button("Generate Plan", type="primary", disabled=(not destination.strip()) or (not allowed))
with col2:
    regen = st.button("Regenerate Suggestions")
with col3:
    clear = st.button("Clear")

if "seed" not in st.session_state:
    st.session_state.seed = 42
if clear:
    st.session_state.pop("results", None)

# ======================= Orchestrator =======================
def generate():
    if provider == "Offline (rule-based)":
        return offline_trip_plan(destination.strip(), days, difficulty, st.session_state.seed), "offline"
    else:
        try:
            attractions, hikes = call_openai(
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
                temperature=temp,
                max_tokens=max_tokens
            )
            return (attractions, hikes), "openai"
        except Exception as e:
            st.error(f"OpenAI error: {e}. Falling back to Offline mode.")
            return offline_trip_plan(destination.strip(), days, difficulty, st.session_state.seed), "offline-fallback"

# ======================= Actions =======================
if (gen or regen) and destination.strip():
    # Double-check RL just before the call
    allowed, reason, _ = can_call_now()
    if not allowed:
        st.warning(reason)
    else:
        (attractions, hikes), mode = generate()
        st.session_state.results = (attractions, hikes, mode)
        record_successful_call()
        if regen:
            st.session_state.seed += 7

# ======================= Results =======================
if "results" in st.session_state:
    attractions, hikes, mode = st.session_state.results
    st.caption(f"Mode: {mode}")

    brand_h2(f"Trip Plan: {destination} · {days} days", brand)
    st.write(
        f"Pace: {travel_style} | Budget: {budget} | Season: {season} | "
        f"Interests: {', '.join(interests) if interests else 'Any'} | "
        f"Hike difficulty: {difficulty}"
    )

    brand_h2("Top Attractions", brand)
    if not attractions:
        st.info("No attractions found.")
    else:
        cols = st.columns(2)
        for i, a in enumerate(attractions):
            with cols[i % 2]:
                links = [
                    ("View on Maps", f"https://www.google.com/maps/search/{a.replace(' ', '+')}"),
                    ("Web Search", f"https://www.google.com/search?q={a.replace(' ', '+')}")
                ]
                section_card(a, links=links)

    brand_h2("Top Hiking Trails", brand)
    if not hikes:
        st.info("No hiking trails found.")
    else:
        cols = st.columns(2)
        for i, h in enumerate(hikes):
            with cols[i % 2]:
                title = h.name
                subtitle = f"Difficulty: {h.difficulty} · Distance: {h.distance} · Route: {h.type}"
                links = [
                    ("Trailhead Maps", f"https://www.google.com/maps/search/{title.replace(' ', '+')}"),
                    ("Trail Info", f"https://www.google.com/search?q={title.replace(' ', '+')}")
                ]
                section_card(title, subtitle_html=subtitle, links=links)

else:
    st.info("Enter a destination and click Generate Plan.")



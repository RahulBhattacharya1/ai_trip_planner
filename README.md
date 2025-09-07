# AI Trip Planner

An interactive Streamlit app that generates **Top Attractions** and **Top Hiking Trails** for your chosen destination.  
It mirrors the same add-ons and sidebar features as your AI Text-to-Slide project — with **provider switch (OpenAI / Offline), model picker, brand color theme, creativity control, and token limits**.

---

## Features
- **Destination input** with optional filters:
  - Season, travel pace, budget, interests
  - Accessibility preferences (kid-friendly, elder-friendly, wheelchair accessible)
  - Hiking difficulty
- **Two modes of operation**:
  - **Offline (rule-based)**: Generates sample results without any API calls (useful for demo/testing).
  - **OpenAI mode**: Calls OpenAI GPT models (gpt-4o, gpt-4.1-mini, etc.) to generate real recommendations.
- **Customizable UI**:
  - Brand color picker
  - Creativity/temperature slider
  - Token length control
- **Outputs**:
  - Two sections: **Top Attractions** and **Top Hiking Trails**
  - Rich cards with Google Maps and web search links
  - One-click Markdown export

---

## Quickstart

1. **Clone or download** this repo.

2. **Install dependencies** (preferably inside a virtual environment):
   ```bash
   pip install -r requirements.txt

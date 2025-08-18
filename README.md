# Perishable Lot-Sizing — Generator + Compact MIP (LEFO)

This is the **old setup** you asked for:
- Generate instances exactly like before (scalar/range setup, per-item perishability ranges, optional item caps, etc.)
- Solve **only** with the compact MIP using your **LEFO (freshest-first)** constraints from LaTeX.
- Simple Streamlit UI.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# ensure Gurobi is installed and licensed on this machine
streamlit run streamlit_app.py
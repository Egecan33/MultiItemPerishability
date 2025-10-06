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
caffeinate -idm streamlit run streamlit_app.py


(.venv) (base) egecanaktan@192 MultiItemPerishability % python parse_demands_and_insert.py \
  --demand-lib-root /Users/egecanaktan/github_repositories/MultiItemPerishability/demands_lib \
  --create-all \
  --instances-per-class 5 \
  --L-list "1,2,3,4,5,6,7,8,9,10,11,12" \
  --synth-variants-per-bucket 15 \
  --pick first \
  -y



python report.py --csv /Users/egecanaktan/github_repositories/MultiItemPerishability/batch11.csv --out out --xlsx
# Perishable Lot-Sizing (Freshest-First MIP) — Toolkit

This repo contains:
- `streamlit_app.py`: a UI to generate problem instances (JSON) using your generator knobs.
- `mip/generator.py`: instance generator (exactly the policy you liked).
- `mip/solver_mip_lefo.py`: compact MIP that matches your LaTeX + PDF (freshest-first via LEFO).
- `db/supabase_schema.sql`: tables for instances, runs, solutions (x,y), metrics.
- `db/runner_supabase.py`: pull next run request from Supabase, solve, push artifacts/rows back.

## Quickstart

1) **Python & dependencies**

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install --upgrade pip
pip install streamlit gurobipy numpy pandas pydantic python-dotenv supabase

Gurobi: install Gurobi + license (grbgetkey, set GRB_LICENSE_FILE). The Python package above just binds to your local Gurobi install.


awesome — i pulled everything together into a clean, drop-in kit:
	•	a Streamlit app to generate instances (using your preferred generator logic).
	•	a Gurobi MIP solver that exactly implements your new LaTeX (incl. the freshest-first LEFO constraints, tight μ_{it}, warehouse capacity, etc.).
	•	an optional Supabase schema + runner so you can schedule/control runs and store results, with example SQL + Python.

Below are the files. You can copy-paste them as-is into a repo (suggested layout shown in the README).

⸻

README.md

# Perishable Lot-Sizing (Freshest-First MIP) — Toolkit

This repo contains:
- `streamlit_app.py`: a UI to generate problem instances (JSON) using your generator knobs.
- `mip/generator.py`: instance generator (exactly the policy you liked).
- `mip/solver_mip_lefo.py`: compact MIP that matches your LaTeX + PDF (freshest-first via LEFO).
- `db/supabase_schema.sql`: tables for instances, runs, solutions (x,y), metrics.
- `db/runner_supabase.py`: pull next run request from Supabase, solve, push artifacts/rows back.

## Quickstart

1) **Python & dependencies**

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install --upgrade pip
pip install streamlit gurobipy numpy pandas pydantic python-dotenv supabase

Gurobi: install Gurobi + license (grbgetkey, set GRB_LICENSE_FILE). The Python package above just binds to your local Gurobi install.

	2.	Generate an instance (via UI)

streamlit run streamlit_app.py

	•	Set PERIOD, #items, demand ranges, setup options, per-item perishability ranges, etc.
	•	Click Generate then Save JSON. It writes last_instance.json (or a name you choose).

	3.	Solve the MIP (Freshest-First)

python -m mip.solver_mip_lefo --instance last_instance.json \
  --time-limit 3600 --mipgap 0.01 --outdir mip_results

Artifacts:
	•	summary.json, metrics.txt, capacity.csv, model.lp
	•	x_nonzero.csv (X_{itu}>0), y_setups.csv (Y_{it}), orders.txt
	•	copy of the instance JSON

	4.	(Optional) Supabase

	•	Create a project, set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or anon key) in .env
	•	Run schema:

psql "<YOUR_POSTGRES_CONNECTION_URL>" -f db/supabase_schema.sql


	•	Start the runner:

python db/runner_supabase.py


	•	Insert a run request (see SQL examples at bottom of db/supabase_schema.sql) — the runner will pick it up, solve, and write rows.


# Warfarin INR Research AI — Synthetic Software Demonstration

**No INR prediction is performed.** A synthetic 7-day *arbitrary teaching score* is computed to demonstrate an AI agent architecture. It is not a medical prediction or a replacement for laboratory/point-of-care INR measurement. Never change anticoagulant dosing, monitoring, or care based on this tool.

## Modules

- `agent/schemas.py`: input validation of warfarin exposure, patient characteristics, nutrition/lifestyle, medicines, illnesses, lab context, pharmacogenetics, INR history, and monitoring context.
- `agent/features.py`: feature transformations and simple medication keyword flags.
- `data/synthetic_data.py`: artificial training examples and arbitrary target generator.
- `models/train.py`: Random Forest, held-out synthetic MAE/RMSE and empirical 90th percentile absolute errors.
- `agent/engine.py`: orchestrates input, model, warnings, global feature importance, and synthetic uncertainty indication.
- `api/main.py`: FastAPI `/health`, `/demo/predict`.
- `dashboard/app.py`: Streamlit interface with editable demonstration JSON.
- `tests/test_agent.py`: validation/API/safety tests.

## Run

Python 3.10+ recommended. From this project directory:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m data.synthetic_data
python -m models.train
python run_sample.py
pytest -q
uvicorn api.main:app --reload
```

FastAPI interactive docs: `http://127.0.0.1:8000/docs`
In a separate terminal: `streamlit run dashboard/app.py`

For API sample request:
```bash
curl -X POST http://127.0.0.1:8000/demo/predict -H 'content-type: application/json' --data-binary @sample_input.json
```

## Scope and limitations

The synthetic target is **deliberately not a medically derived INR** and does not convey risk, treatment, or dose requirements. The medication keyword screening is incomplete and cannot exclude interactions. The input schema includes many researched factor categories but only some have variation in the arbitrary synthetic target; mere inclusion of a field does not prove predictive power. Global feature importance is neither a per-patient explanation nor causal evidence. An empirical held-out synthetic error band is not a validated clinical interval. The dataset contains no real patient observations and no external validation. Date-stamped real measured INR history supplied through the demonstration only changes artificial features, not an actual INR estimate.

An eventual clinical research model requires ethical and lawful collection of longitudinal measured INR outcomes, time-ordered and patient-separated evaluation, external validation, bias/subgroup analysis, reproducible governance, calibration, clinical oversight, privacy/security controls, regulatory review, and a studied human-in-the-loop workflow.

The prototype stores no patient data by default. Do not submit sensitive identifiable patient records to this development demo.

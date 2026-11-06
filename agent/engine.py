from pathlib import Path
import joblib
from agent.features import FEATURES,to_features,KEYWORDS
from agent.schemas import ResearchInput

MODEL_PATH=Path(__file__).resolve().parents[1]/'models'/'artifacts'/'demo_model.joblib'
WARNING=('EDUCATIONAL SYNTHETIC SCORE ONLY. This output is NOT an INR estimate, '
         'does not predict bleeding/clotting risk, and must NOT guide warfarin dosing or testing.')

def evaluate(payload:ResearchInput):
    if not MODEL_PATH.exists():
        raise RuntimeError('Demo model not trained. Run python -m models.train first.')
    bundle=joblib.load(MODEL_PATH)
    if bundle['features']!=FEATURES:raise RuntimeError('Model schema mismatch')
    row=to_features(payload)
    import pandas as pd
    val=float(bundle['model'].predict(pd.DataFrame([row],columns=FEATURES))[0]); q=bundle['q90']
    flags=[]
    if payload.medication_names:
        matched=[s for s in payload.medication_names if any(k in s.lower() for k in KEYWORDS)]
        if matched:flags.append('Medication names warrant pharmacist/clinician interaction review: '+', '.join(matched))
    if payload.missed_doses_7d or payload.extra_doses_7d or payload.dose_interrupted:
        flags.append('Recent dosing inconsistency reported; seek clinical guidance, not algorithmic adjustment.')
    if payload.acute_infection or payload.diarrhea_or_vomiting or payload.low_appetite:
        flags.append('Acute illness or nutrition change reported; consider contacting anticoagulation team.')
    if not payload.recent_inr:flags.append('No prior measured INR observations supplied.')
    if payload.pregnancy:flags.append('Pregnancy indicated: urgent review with treating medical team is warranted for warfarin exposure.')
    # Not causal explanations: RF impurity global feature importances, descriptive only.
    indices=sorted(range(len(FEATURES)),key=lambda i:bundle['model'].feature_importances_[i],reverse=True)[:6]
    return {'horizon_days':7,'output_name':'synthetic_teaching_score','synthetic_teaching_score':round(val,2),
       'illustrative_holdout_error_band':[round(val-q,2),round(val+q,2)],
       'global_model_feature_importance':[{'feature':FEATURES[i], 'importance':round(float(bundle['model'].feature_importances_[i]),4)} for i in indices],
       'review_flags':flags,'warning':WARNING,'clinical_inr_estimate':None,'dose_recommendation':None}

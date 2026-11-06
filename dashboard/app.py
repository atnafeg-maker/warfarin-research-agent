import streamlit as st
from pydantic import ValidationError
from agent.schemas import ResearchInput
from agent.engine import evaluate,WARNING
st.set_page_config(page_title='INR Research — Synthetic Demo',layout='centered')
st.title('Warfarin research agent — synthetic demonstration')
st.error(WARNING)
st.write('The synthetic score is an arbitrary software demonstration, NOT an INR value.')
example={
 'age_years':55,'weight_kg':75,'height_cm':172,'sex':'male','indication':'mechanical_valve',
 'daily_dose_mg':5,'prior_daily_dose_mg':5,'therapy_days':200,
 'missed_doses_7d':0,'vitamin_k_change':'stable',
 'medication_names':[],'recent_inr':[{'date':'2026-10-20','measured_inr':2.7},{'date':'2026-10-27','measured_inr':2.8}]
}
st.caption('Edit the JSON to explore the software pipeline; do not enter identifying health information.')
import json
raw=st.text_area('Demonstration input JSON',json.dumps(example,indent=2),height=350)
if st.button('Run synthetic demonstration'):
    try:
        out=evaluate(ResearchInput.model_validate_json(raw))
        st.metric('Artificial teaching score (not INR)',out['synthetic_teaching_score'])
        st.write('Illustrative synthetic error band:',out['illustrative_holdout_error_band'])
        st.write('Global model feature importance (not patient-specific cause):')
        st.dataframe(out['global_model_feature_importance'])
        st.write('Review flags:',out['review_flags'] or ['None'])
        st.json(out)
    except (ValidationError,ValueError,RuntimeError) as e:st.error(str(e))

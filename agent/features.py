import numpy as np
from agent.schemas import ResearchInput

FEATURES = [
 'age_years','weight_kg','height_cm','sex_male','ind_mechanical','daily_dose_mg','prior_daily_dose_mg',
 'therapy_days','missed_doses_7d','extra_doses_7d','delayed_doses_7d',
 'dose_interrupted','bridge_therapy','administration_time_changed','vitamin_k_change',
 'meals_irregular','low_appetite','alcohol_change','supplement_change','medication_change',
 'liver_disease','kidney_disease','heart_failure_change','thyroid_condition','acute_infection',
 'diarrhea_or_vomiting','malabsorption','cancer_therapy','recent_surgery','pregnancy',
 'albumin_g_dl','egfr','bilirubin_mg_dl','hemoglobin_g_dl','platelet_count_k','tsh',
 'cyp2c9_known','vkorc1_known','cyp4f2_known','prior_bleed','prior_clot','recent_inr_count',
 'recent_inr_latest','recent_inr_mean','recent_inr_trend','med_count','interaction_keyword_count'
]
KEYWORDS=('amiodarone','metronidazole','trimethoprim','fluconazole','rifampin',
          'carbamazepine','phenytoin','acetaminophen','aspirin','ibuprofen','naproxen')

def to_features(inp:ResearchInput):
    o=inp.model_dump()
    d={k:float(o.get(k) or 0) for k in FEATURES if k in o and isinstance(o.get(k),(int,float,bool,type(None)))}
    d.update(sex_male=float(inp.sex=='male'),ind_mechanical=float(inp.indication=='mechanical_valve'),
             vitamin_k_change={'decreased':-1,'stable':0,'increased':1}[inp.vitamin_k_change],
             cyp2c9_known=float(bool(inp.cyp2c9)),vkorc1_known=float(bool(inp.vkorc1)),
             cyp4f2_known=float(bool(inp.cyp4f2)))
    v=[x.measured_inr for x in inp.recent_inr]
    d.update(recent_inr_count=len(v),recent_inr_latest=v[-1] if v else 0,
             recent_inr_mean=float(np.mean(v)) if v else 0,
             recent_inr_trend=(v[-1]-v[-2]) if len(v)>1 else 0,
             med_count=len(inp.medication_names),
             interaction_keyword_count=sum(any(k in s.lower() for k in KEYWORDS) for s in inp.medication_names))
    return [float(d.get(k,0)) for k in FEATURES]

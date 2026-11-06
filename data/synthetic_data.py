"""Synthetic arbitrary teaching-score generator; NOT an INR simulator."""
import numpy as np
import pandas as pd
from agent.features import FEATURES

def make_data(n=2400,seed=2026):
    rng=np.random.default_rng(seed)
    cols={k:np.zeros(n) for k in FEATURES}
    cols['age_years']=rng.integers(20,90,n)
    cols['weight_kg']=rng.uniform(45,125,n)
    cols['height_cm']=rng.uniform(145,200,n)
    cols['daily_dose_mg']=rng.uniform(1,12,n)
    cols['prior_daily_dose_mg']=cols['daily_dose_mg']+rng.normal(0,1,n)
    cols['therapy_days']=rng.integers(1,2000,n)
    cols['vitamin_k_change']=rng.choice([-1,0,1],n,p=[.2,.6,.2])
    cols['recent_inr_count']=rng.integers(0,5,n)
    cols['recent_inr_latest']=rng.uniform(1.5,4.0,n)
    cols['recent_inr_mean']=rng.uniform(1.5,4.0,n)
    cols['recent_inr_trend']=rng.normal(0,.3,n)
    cols['missed_doses_7d']=rng.binomial(3,.12,n)
    cols['extra_doses_7d']=rng.binomial(2,.035,n)
    cols['med_count']=rng.integers(0,5,n)
    cols['interaction_keyword_count']=rng.binomial(2,.12,n)
    for k in FEATURES:
        if k not in ('age_years','weight_kg','height_cm','daily_dose_mg','prior_daily_dose_mg','therapy_days','vitamin_k_change','recent_inr_count','recent_inr_latest','recent_inr_mean','recent_inr_trend','missed_doses_7d','extra_doses_7d','med_count','interaction_keyword_count'):
            if k in ('albumin_g_dl','hemoglobin_g_dl','egfr','platelet_count_k','tsh','bilirubin_mg_dl'):
                ranges={'albumin_g_dl':(2,5),'hemoglobin_g_dl':(9,17),'egfr':(20,120),'platelet_count_k':(100,400),'tsh':(.5,8),'bilirubin_mg_dl':(.2,2)}
                cols[k]=rng.uniform(*ranges[k],n)
            else: cols[k]=rng.binomial(1,.13,n)
    # Deliberately fabricated learning function with arbitrary weights, not biologically validated.
    score=(50+1.8*cols['recent_inr_latest']+.35*cols['daily_dose_mg']
           -1.5*cols['missed_doses_7d']+1.5*cols['extra_doses_7d']
           -1.2*cols['vitamin_k_change']+.8*cols['medication_change']
           +.4*cols['acute_infection']+rng.normal(0,2,n))
    return pd.DataFrame(cols)[FEATURES],score

if __name__=='__main__':
    from pathlib import Path
    X,y=make_data();d=Path(__file__).resolve().parent
    X.assign(teaching_score_7d=y).to_csv(d/'synthetic_training.csv',index=False)
    print('Wrote',len(X),'synthetic records')

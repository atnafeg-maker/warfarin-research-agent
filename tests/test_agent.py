from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from agent.schemas import ResearchInput
from agent.features import FEATURES,to_features
from api.main import app
from models.train import train

BASE={'age_years':55,'weight_kg':75,'height_cm':172,'daily_dose_mg':5,
      'prior_daily_dose_mg':5,'therapy_days':100}

def test_validation():
    with pytest.raises(Exception):ResearchInput(**(BASE|{'age_years':-2}))
    with pytest.raises(Exception):ResearchInput(**(BASE|{'target_low':4,'target_high':3}))

def test_feature_shape():
    assert len(to_features(ResearchInput(**BASE)))==len(FEATURES)

def test_training_and_api():
    metrics=train();assert metrics['test_MAE']>=0
    client=TestClient(app)
    assert client.get('/health').status_code==200
    r=client.post('/demo/predict',json=BASE)
    assert r.status_code==200,r.text
    result=r.json()
    assert result['clinical_inr_estimate'] is None
    assert result['dose_recommendation'] is None
    assert result['output_name']=='synthetic_teaching_score'
    assert 'NOT an INR' in result['warning']
    assert len(result['global_model_feature_importance'])==6

def test_bad_input_api():
    assert TestClient(app).post('/demo/predict',json={'daily_dose_mg':5}).status_code==422

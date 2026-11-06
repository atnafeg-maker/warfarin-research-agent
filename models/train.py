from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from data.synthetic_data import make_data
from agent.features import FEATURES
MODEL_DIR=Path(__file__).resolve().parent/'artifacts'

def train():
    X,y=make_data()
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=12)
    model=RandomForestRegressor(n_estimators=100,min_samples_leaf=5,random_state=12,n_jobs=-1)
    model.fit(Xtr,ytr); preds=model.predict(Xte)
    mae=float(mean_absolute_error(yte,preds));rmse=float(np.sqrt(mean_squared_error(yte,preds)))
    q=float(np.quantile(abs(yte-preds),.9))
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({'model':model,'features':FEATURES,'q90':q},MODEL_DIR/'demo_model.joblib')
    report={'target':'synthetic arbitrary teaching score (NOT INR)','n_train':len(Xtr),'n_test':len(Xte),
        'test_MAE':mae,'test_RMSE':rmse,'holdout_absolute_error_q90':q,
        'caution':'One synthetic split; q90 is not a medically valid prediction interval or external calibration.'}
    (MODEL_DIR/'metrics.json').write_text(json.dumps(report,indent=2))
    return report
if __name__=='__main__':print(json.dumps(train(),indent=2))

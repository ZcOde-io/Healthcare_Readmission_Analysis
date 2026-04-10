from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import joblib, json, numpy as np, os

Base = declarative_base()
engine = create_engine("sqlite:///readmission.db", connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)

class Prediction(Base):
    __tablename__ = "predictions"
    id               = Column(Integer, primary_key=True, index=True)
    timestamp        = Column(DateTime, default=datetime.utcnow)
    age              = Column(Integer)
    time_in_hospital = Column(Integer)
    num_medications  = Column(Integer)
    number_inpatient = Column(Integer)
    number_emergency = Column(Integer)
    HbA1c_result     = Column(String)
    admission_type   = Column(String)
    diabetesMed      = Column(String)
    risk_score       = Column(Float)
    prediction       = Column(String)

Base.metadata.create_all(engine)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
model     = joblib.load(f"{MODEL_DIR}/best_model.pkl")
scaler    = joblib.load(f"{MODEL_DIR}/scaler.pkl")
feat_cols = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
with open(f"{MODEL_DIR}/metrics.json") as f:
    metrics = json.load(f)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
app       = FastAPI(title="Diabetes Readmission Predictor")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR,"static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR,"templates"))

AGE_MAP = {'[20-30)':25,'[30-40)':35,'[40-50)':45,'[50-60)':55,
           '[60-70)':65,'[70-80)':75,'[80-90)':85,'[90-100)':95}
CAT_ENC = {
    'race':                 {'Caucasian':0,'AfricanAmerican':1,'Hispanic':2,'Asian':3,'Other':4},
    'gender':               {'Female':0,'Male':1},
    'admission_type':       {'Elective':0,'Emergency':1,'Other':2,'Urgent':3},
    'discharge_disposition':{'AMA':0,'Home':1,'Other':2,'Rehab':3,'SNF':4},
    'HbA1c_result':         {'>7':0,'>8':1,'None':2,'Normal':3},
    'insulin':              {'Down':0,'No':1,'Steady':2,'Up':3,'Yes':4},
    'diabetesMed':          {'No':0,'Yes':1},
    'change':               {'No':0,'Yes':1},
}

def build_feature_vector(form):
    age_num       = AGE_MAP.get(form.get('age','[60-70)'), 65)
    time_in_hosp  = int(form.get('time_in_hospital',5))
    num_lab       = int(form.get('num_lab_procedures',40))
    num_proc      = int(form.get('num_procedures',1))
    num_med       = int(form.get('num_medications',15))
    num_out       = int(form.get('number_outpatient',0))
    num_em        = int(form.get('number_emergency',0))
    num_in        = int(form.get('number_inpatient',0))
    num_diag      = int(form.get('number_diagnoses',5))
    high_risk     = int(num_in >= 2 or num_em >= 1)
    med_intensity = num_med * num_proc
    total_visits  = num_out + num_em + num_in
    cats = {c+'_enc': CAT_ENC[c].get(str(form.get(c,'')), 0) for c in CAT_ENC}
    row = [age_num,time_in_hosp,num_lab,num_proc,num_med,num_out,num_em,num_in,
           num_diag,high_risk,med_intensity,total_visits]
    for c in ['race','gender','admission_type','discharge_disposition',
              'HbA1c_result','insulin','diabetesMed','change']:
        row.append(cats[c+'_enc'])
    return np.array(row).reshape(1,-1)

def tmpl(req, name, ctx):
    try:
        return templates.TemplateResponse(req, name, ctx)
    except TypeError:
        ctx['request'] = req
        return templates.TemplateResponse(name, ctx)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    db = Session()
    recent = db.query(Prediction).order_by(Prediction.id.desc()).limit(10).all()
    db.close()
    return tmpl(request,"index.html",{"metrics":metrics,"recent":recent,
                                       "best_model":metrics['best_model'],"result":None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: str = Form(...),
    time_in_hospital: int = Form(...),
    num_lab_procedures: int = Form(...),
    num_procedures: int = Form(...),
    num_medications: int = Form(...),
    number_outpatient: int = Form(...),
    number_emergency: int = Form(...),
    number_inpatient: int = Form(...),
    number_diagnoses: int = Form(...),
    race: str = Form(...),
    gender: str = Form(...),
    admission_type: str = Form(...),
    discharge_disposition: str = Form(...),
    HbA1c_result: str = Form(...),
    insulin: str = Form(...),
    diabetesMed: str = Form(...),
    change: str = Form(...),
):
    form  = {k:v for k,v in locals().items() if k!='request'}
    X     = build_feature_vector(form)
    prob  = float(model.predict_proba(X)[0,1])
    label = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
    db = Session()
    db.add(Prediction(age=AGE_MAP.get(age,65),time_in_hospital=time_in_hospital,
        num_medications=num_medications,number_inpatient=number_inpatient,
        number_emergency=number_emergency,HbA1c_result=HbA1c_result,
        admission_type=admission_type,diabetesMed=diabetesMed,
        risk_score=round(prob,4),prediction=label))
    db.commit()
    recent = db.query(Prediction).order_by(Prediction.id.desc()).limit(10).all()
    db.close()
    return tmpl(request,"index.html",{"metrics":metrics,"recent":recent,
        "best_model":metrics['best_model'],
        "result":{"label":label,"prob":round(prob*100,1),"form":form}})

@app.get("/history", response_class=HTMLResponse)
async def history(request: Request):
    db = Session()
    all_preds = db.query(Prediction).order_by(Prediction.id.desc()).all()
    db.close()
    return tmpl(request,"history.html",{"predictions":all_preds})

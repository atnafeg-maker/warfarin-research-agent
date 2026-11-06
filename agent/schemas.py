from typing import Literal
from pydantic import BaseModel, Field, model_validator

class Observation(BaseModel):
    date: str = Field(description="ISO date, pre-prediction only")
    measured_inr: float = Field(ge=0.5, le=20)

class ResearchInput(BaseModel):
    age_years: int = Field(ge=18,le=110)
    weight_kg: float = Field(gt=20,le=350)
    height_cm: float = Field(ge=100,le=240)
    sex: Literal['female','male','other','unspecified'] = 'unspecified'
    indication: Literal['mechanical_valve','afib','dvt','pe','other'] = 'other'
    daily_dose_mg: float = Field(ge=0,le=30)
    prior_daily_dose_mg: float = Field(ge=0,le=30)
    therapy_days: int = Field(ge=0,le=20000)
    missed_doses_7d: int = Field(ge=0,le=7,default=0)
    extra_doses_7d: int = Field(ge=0,le=7,default=0)
    delayed_doses_7d: int = Field(ge=0,le=7,default=0)
    dose_interrupted: bool = False
    bridge_therapy: bool = False
    administration_time_changed: bool = False
    vitamin_k_change: Literal['decreased','stable','increased'] = 'stable'
    meals_irregular: bool = False
    low_appetite: bool = False
    alcohol_change: bool = False
    supplement_change: bool = False
    medication_names: list[str] = Field(default_factory=list)
    medication_change: bool = False
    liver_disease: bool = False
    kidney_disease: bool = False
    heart_failure_change: bool = False
    thyroid_condition: bool = False
    acute_infection: bool = False
    diarrhea_or_vomiting: bool = False
    malabsorption: bool = False
    cancer_therapy: bool = False
    recent_surgery: bool = False
    pregnancy: bool = False
    albumin_g_dl: float | None = Field(default=None,ge=1,le=6)
    egfr: float | None = Field(default=None,ge=0,le=150)
    bilirubin_mg_dl: float | None = Field(default=None,ge=0,le=30)
    hemoglobin_g_dl: float | None = Field(default=None,ge=3,le=22)
    platelet_count_k: float | None = Field(default=None,ge=1,le=1000)
    tsh: float | None = Field(default=None,ge=0,le=100)
    cyp2c9: str | None = None
    vkorc1: str | None = None
    cyp4f2: str | None = None
    prior_bleed: bool = False
    prior_clot: bool = False
    recent_inr: list[Observation] = Field(default_factory=list,max_length=20)
    target_low: float = Field(ge=1,le=5,default=2)
    target_high: float = Field(ge=1,le=6,default=3)
    prediction_horizon_days: Literal[7] = 7

    @model_validator(mode='after')
    def check(self):
        if self.target_low >= self.target_high: raise ValueError('target_low must be below target_high')
        from datetime import date
        days=[date.fromisoformat(o.date) for o in self.recent_inr]
        if len(set(days)) != len(days): raise ValueError('Duplicate INR dates')
        if days != sorted(days): raise ValueError('INR history must be in chronological order')
        if any(d>date.today() for d in days): raise ValueError('Future INR observation not accepted')
        return self

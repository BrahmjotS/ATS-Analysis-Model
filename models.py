from pydantic import BaseModel
from typing import Literal


class ResumeAnalysis(BaseModel):
    overall_score: int
    ats_friendly: Literal["Yes", "Intermediate", "No"]
    strengths: str
    x_factor: str
    weaknesses: str
    fixes: list[str]
    suggested_roles: list[str]
    ats_keywords_to_add: list[str]


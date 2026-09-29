import json

import pytest

VALID = {
    "title": "Senior Backend Engineer",
    "company": "Acme Corp",
    "location": "Berlin",
    "work_mode": "hybrid",
    "employment_type": "full_time",
    "seniority": "senior",
    "salary": {"min": 70000, "max": 90000, "currency": "EUR", "period": "year"},
    "required_skills": ["Python", " python ", "Postgres"],
    "years_experience_min": 5,
    "apply_url": "https://acme.example/jobs/1",
    "confidence": 0.9,
}


def as_json(**overrides) -> str:
    d = {**VALID, **overrides}
    return json.dumps({k: v for k, v in d.items() if v is not ...})


@pytest.fixture
def valid_json() -> str:
    return as_json()

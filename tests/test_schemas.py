import pytest
from pydantic import ValidationError

from agent.schemas import CurrencyNormalization, JobPosting, Salary

from .conftest import VALID


def test_valid_posting_normalizes_skills():
    p = JobPosting.model_validate(VALID)
    assert p.required_skills == ["postgres", "python"]


def test_salary_min_above_max_is_cross_field():
    with pytest.raises(ValidationError) as ei:
        Salary(min=100, max=50)
    assert ei.value.errors()[0]["type"] == "cross_field"


def test_currency_must_be_iso_code():
    with pytest.raises(ValidationError):
        Salary(currency="usd")


def test_currency_normalization_consistency():
    with pytest.raises(ValidationError):
        CurrencyNormalization(raw="$", code=None, recognized=True)
    assert CurrencyNormalization(raw="?", code=None, recognized=False)

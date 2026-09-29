import pytest

from agent.extractor import extract
from agent.llm import get_llm

POSTING = """\
Acme Robotics is hiring a Senior Python Engineer (remote, EU time zones).
Full-time. €80k–€100k per year. 5+ years with Python, Postgres and AWS.
Apply: https://acme-robotics.example/careers/42
"""


@pytest.mark.live
def test_live_extraction():
    res = extract(POSTING, get_llm(), logger=False)
    assert res.ok, res.last_error
    assert res.posting.work_mode == "remote"
    assert res.posting.salary and res.posting.salary.currency == "EUR"

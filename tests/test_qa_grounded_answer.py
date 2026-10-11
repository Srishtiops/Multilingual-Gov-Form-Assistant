import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gov_form_assistant.qa import FormQA


def test_extract_grounded_answer_finds_named_field_value():
    context = """
    Applicant Name: Rohan Sharma
    Date of Birth: 12/05/1990
    Address: 42 Main Road, Jaipur
    """

    answer = FormQA._extract_grounded_answer("What is the applicant's name?", context)

    assert answer is not None
    assert "Rohan Sharma" in answer


def test_extract_grounded_answer_returns_none_when_not_present():
    context = """
    Please attach a passport size photo.
    Submit the application with required documents.
    """

    answer = FormQA._extract_grounded_answer("What is the applicant's date of birth?", context)

    assert answer is None

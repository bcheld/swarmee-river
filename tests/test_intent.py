from swarmee_river.planning import classify_intent


def test_classify_intent_work():
    assert classify_intent("Fix the failing tests in swarmee.py") == "work"


def test_classify_intent_info():
    assert classify_intent("List the American presidents in chronological order") == "info"

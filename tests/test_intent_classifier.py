import os
import builtins
import types
import pytest

from bigq_ai_app.core.intent_classifier import classify_intent, generate


class DummyResp:
    def __init__(self, text: str):
        self.text = text


def test_classify_intent_analytics(monkeypatch):
    # Monkeypatch generate() to avoid network calls
    monkeypatch.setattr(
        "bigq_ai_app.core.intent_classifier.generate",
        lambda prompt, **kwargs: "analytics",
    )

    result = classify_intent("What is the latest transaction count?")
    assert result == "analytics"


def test_classify_intent_non_analytics(monkeypatch):
    monkeypatch.setattr(
        "bigq_ai_app.core.intent_classifier.generate",
        lambda prompt, **kwargs: "non_analytics",
    )

    result = classify_intent("What is the capital of Italy?")
    assert result == "non_analytics"


def test_classify_intent_noise_returns_non_analytics(monkeypatch):
    monkeypatch.setattr(
        "bigq_ai_app.core.intent_classifier.generate",
        lambda prompt, **kwargs: "I think it's not analytics",
    )
    assert classify_intent("Tell me a joke") == "non_analytics"

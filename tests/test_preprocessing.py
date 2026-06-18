import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re


def clean_narrative(text):
    """Same cleaning function from Task 1"""
    text = str(text)
    if text.startswith("b'") or text.startswith('b"'):
        text = text[2:-1]
    text = text.lower()
    boilerplate_patterns = [
        r'i am writing to file a complaint.*?\.',
        r'dear cfpb.*?\,',
        r'xxxx',
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\{\$[\d,]+\.?\d*\}', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def test_clean_narrative_lowercase():
    """Test that cleaning lowercases text"""
    result = clean_narrative("THIS IS A COMPLAINT")
    assert result == "this is a complaint"


def test_clean_narrative_removes_xxxx():
    """Test that XXXX placeholders are removed"""
    result = clean_narrative("My XXXX card was declined")
    assert "xxxx" not in result


def test_clean_narrative_removes_special_chars():
    """Test that special characters are removed"""
    result = clean_narrative("Charged $500!! without#warning")
    assert "$" not in result
    assert "!" not in result
    assert "#" not in result


def test_clean_narrative_removes_boilerplate():
    """Test that boilerplate phrases are removed"""
    result = clean_narrative("Dear CFPB, I have a problem with my card")
    assert "dear cfpb" not in result


def test_clean_narrative_strips_whitespace():
    """Test that extra spaces are removed"""
    result = clean_narrative("too   many    spaces")
    assert "  " not in result


def test_clean_narrative_handles_empty():
    """Test that empty string is handled safely"""
    result = clean_narrative("")
    assert result == ""


def test_clean_narrative_handles_bytes_prefix():
    """Test that bytes prefix b' is removed"""
    result = clean_narrative("b'some complaint text'")
    assert not result.startswith("b")
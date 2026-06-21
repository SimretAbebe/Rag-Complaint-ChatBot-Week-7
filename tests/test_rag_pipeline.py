import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generator import build_prompt, PROMPT_TEMPLATE
from src.retriever import format_context


def test_prompt_contains_context():
    """Test that build_prompt inserts context correctly"""
    context = "Customer complained about credit card fees"
    question = "Why are customers unhappy?"
    prompt = build_prompt(question, context)
    assert context in prompt
    assert question in prompt


def test_prompt_contains_question():
    """Test that build_prompt inserts question correctly"""
    prompt = build_prompt("test question", "test context")
    assert "test question" in prompt
    assert "test context" in prompt


def test_format_context_returns_string():
    """Test that format_context returns a string"""
    chunks = [
        {
            "text": "My card was declined",
            "metadata": {
                "product_category": "Credit Card",
                "issue": "Card declined",
                "company": "CITIBANK"
            },
            "similarity_score": 0.92
        }
    ]
    result = format_context(chunks)
    assert isinstance(result, str)
    assert "Credit Card" in result
    assert "CITIBANK" in result


def test_format_context_multiple_chunks():
    """Test that format_context handles multiple chunks"""
    chunks = [
        {
            "text": f"Complaint text {i}",
            "metadata": {
                "product_category": "Credit Card",
                "issue": "Billing dispute",
                "company": "BANK A"
            },
            "similarity_score": 0.9
        }
        for i in range(3)
    ]
    result = format_context(chunks)
    assert "Complaint 1" in result
    assert "Complaint 2" in result
    assert "Complaint 3" in result


def test_format_context_empty_chunks():
    """Test that format_context handles empty list"""
    result = format_context([])
    assert isinstance(result, str)


def test_prompt_template_has_placeholders():
    """Test that prompt template has required placeholders"""
    assert "{context}" in PROMPT_TEMPLATE
    assert "{question}" in PROMPT_TEMPLATE


def test_prompt_template_has_rules():
    """Test that prompt template contains grounding rules"""
    assert "ONLY" in PROMPT_TEMPLATE or "only" in PROMPT_TEMPLATE
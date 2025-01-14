import pytest
from scripts.preprocess_data import preprocess_text

def test_preprocess_text():
    input_text = "Visit https://example.com for free coupons!"
    expected_output = "visit for free coupons"
    assert preprocess_text(input_text) == expected_output

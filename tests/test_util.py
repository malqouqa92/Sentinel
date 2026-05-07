import pytest
from core.util import is_palindrome

def test_simple_palindrome():
    assert is_palindrome("racecar") is True

def test_simple_non_palindrome():
    assert is_palindrome("hello") is False

def test_empty_string():
    assert is_palindrome("") is True

def test_single_char():
    assert is_palindrome("a") is True

def test_two_same():
    assert is_palindrome("aa") is True

def test_two_different():
    assert is_palindrome("ab") is False

def test_mixed_case_default():
    assert is_palindrome("Racecar") is True

def test_mixed_case_sensitive():
    assert is_palindrome("Racecar", ignore_case=False) is False

def test_spaces_ignored_default():
    assert is_palindrome("never odd or even") is True

def test_spaces_not_ignored():
    assert is_palindrome("never odd or even", ignore_spaces=False) is False

def test_classic_phrase():
    assert is_palindrome("A man a plan a canal Panama") is True

def test_numeric_string():
    assert is_palindrome("12321") is True

def test_numeric_non_palindrome():
    assert is_palindrome("12345") is False

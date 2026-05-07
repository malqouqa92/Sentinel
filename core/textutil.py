def count_vowels(s):
    """Return the count of vowels in string s."""
    return sum(1 for ch in s if ch in 'aeiouAEIOU')

"""core/util.py — General-purpose utility functions and data structures for Sentinel OS.

Provides pure-Python helpers (math, string, collection) that are dependency-free
and safe to import from any module without risk of circular imports or VRAM usage.
All functions are synchronous and contain no LLM calls, DB access, or I/O side-effects.
"""

def add(a, b):
    """Return the sum of two numbers."""
    return a + b


def add_integers(a: int, b: int) -> int:
    """Return the sum of two integers.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        The arithmetic sum a + b as an int.

    Raises:
        TypeError: If either argument is not an int.

    Examples:
        >>> add_integers(2, 3)
        5
        >>> add_integers(-1, 1)
        0
        >>> add_integers(0, 0)
        0
    """
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError(
            f"add_integers() requires int arguments, got {type(a).__name__} and {type(b).__name__}"
        )
    return a + b


def sum_two(a: float, b: float) -> float:
    """Return the arithmetic sum of two numbers.

    Args:
        a: First operand.
        b: Second operand.

    Returns:
        The value a + b.

    Examples:
        >>> sum_two(2, 3)
        5
        >>> sum_two(-1, 1)
        0
        >>> sum_two(1.5, 2.5)
        4.0
    """
    return a + b

def subtract(a, b):
    """Return the difference of two numbers (a - b)."""
    return a - b

def square(x):
    """Return the square of x."""
    return x * x

def multiply(a, b):
    """Return the product of two numbers."""
    return a * b

def divide(a, b):
    """Return the quotient of a divided by b. Raises ValueError on division by zero."""
    if b == 0:
        raise ValueError("divide() division by zero is undefined")
    return a / b

def factorial(n):
    """Return n! using recursion. Raises ValueError for negative input."""
    if n < 0:
        raise ValueError("factorial() not defined for negative values")
    if n == 0:
        return 1
    return n * factorial(n - 1)

def clamp(x, lo, hi):
    """Return x bounded between lo and hi inclusive.

    Args:
        x:  The value to clamp.
        lo: Lower bound (inclusive).
        hi: Upper bound (inclusive).

    Returns:
        lo if x < lo, hi if x > hi, otherwise x.

    Raises:
        ValueError: If lo > hi.

    Examples:
        >>> clamp(5, 1, 10)
        5
        >>> clamp(-3, 0, 100)
        0
        >>> clamp(200, 0, 100)
        100
        >>> clamp(0, 0, 0)
        0
    """
    if lo > hi:
        raise ValueError(f"clamp() requires lo <= hi, got lo={lo!r} hi={hi!r}")
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def larger_of(a: float, b: float) -> float:
    """Return the larger of two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        *a* if a >= b, otherwise *b*.

    Examples:
        >>> larger_of(3, 7)
        7
        >>> larger_of(5.5, 5.5)
        5.5
        >>> larger_of(-1, -10)
        -1
    """
    return a if a >= b else b


def is_palindrome(s: str, *, ignore_case: bool = True, ignore_spaces: bool = True) -> bool:
    """Return True if *s* reads the same forwards and backwards.

    Args:
        s:             The string to test.
        ignore_case:   When True (default), 'Racecar' is treated as a palindrome.
        ignore_spaces: When True (default), spaces are stripped before the check
                       so 'never odd or even' is treated as a palindrome.

    Returns:
        bool — True if the cleaned string is a palindrome, False otherwise.

    Examples:
        >>> is_palindrome('racecar')
        True
        >>> is_palindrome('hello')
        False
        >>> is_palindrome('A man a plan a canal Panama')
        True
        >>> is_palindrome('A man a plan a canal Panama', ignore_spaces=False)
        False
    """
    cleaned = s
    if ignore_case:
        cleaned = cleaned.lower()
    if ignore_spaces:
        cleaned = cleaned.replace(' ', '')
    return cleaned == cleaned[::-1]


def merge_dicts(a, b):
    """Return a new dict with all keys from a and b. Keys in b override keys in a."""
    return {**a, **b}

def min_max(lst):
    """Return a (min, max) tuple for the given list. Raises ValueError for empty input."""
    if not lst:
        raise ValueError("min_max() requires a non-empty list")
    return (min(lst), max(lst))

def slugify(text):
    """Return a URL-friendly slug from text (lowercase, hyphens, no special chars)."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    text = re.sub(r'-+', '-', text)
    return text.strip('-')

def chunked(lst, n):
    """Yield successive n-sized sub-lists from lst. Raises ValueError if n < 1."""
    if n < 1:
        raise ValueError("chunked() chunk size n must be >= 1")
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def flatten(lst):
    """Recursively flatten a nested list into a single flat list."""
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def partition(pred, lst):
    """Split lst into two lists based on pred. Returns (yes, no) where yes contains
    items for which pred(item) is True, and no contains items for which it is False."""
    yes, no = [], []
    for item in lst:
        (yes if pred(item) else no).append(item)
    return (yes, no)


class Stack:
    """LIFO stack with push, pop, and peek operations."""

    def __init__(self):
        self._data = []

    def push(self, item):
        """Push item onto the top of the stack."""
        self._data.append(item)

    def pop(self):
        """Remove and return the top item. Raises IndexError if the stack is empty."""
        if not self._data:
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self):
        """Return the top item without removing it. Raises IndexError if the stack is empty."""
        if not self._data:
            raise IndexError("peek at empty stack")
        return self._data[-1]

    def __len__(self):
        return len(self._data)

    def __bool__(self):
        return bool(self._data)

    def __repr__(self):
        return f"Stack({self._data!r})"

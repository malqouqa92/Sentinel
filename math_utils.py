def add(a, b):
    """Add two numbers and return the result."""
    return a + b


def add_numbers(a: float, b: float) -> float:
    """Return the sum of two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The arithmetic sum a + b.
    """
    return a + b


def add_integers(a: int, b: int) -> int:
    """Return the sum of two integers.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        The arithmetic sum a + b as an int.
    """
    return a + b


def larger_of(a, b):
    """Return the larger of two numbers."""
    return a if a >= b else b


def lcm(a: int, b: int) -> int:
    """Return the least common multiple of two positive integers.

    Args:
        a: A positive integer.
        b: A positive integer.

    Returns:
        The smallest positive integer divisible by both a and b.

    Raises:
        ValueError: If either a or b is non-positive.
    """
    if a <= 0 or b <= 0:
        raise ValueError(f"lcm() requires positive integers, got {a} and {b}")
    from math import gcd
    return abs(a * b) // gcd(a, b)


def factorial(n: int) -> int:
    """Return n! (n factorial).

    Args:
        n: A non-negative integer.

    Returns:
        The product 1 * 2 * ... * n.  factorial(0) == 1.

    Raises:
        ValueError: If n is negative.
    """
    if not isinstance(n, int):
        raise TypeError(f"factorial() requires an int, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"factorial() not defined for negative values, got {n}")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

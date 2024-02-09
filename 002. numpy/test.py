def fact(n):
    """Calculates the factorial of a non-negative integer n.

    Args:
        n: The non-negative integer for which to calculate the factorial.

    Returns:
        The factorial of n, or None if n is negative.
    """

    if n < 0:
        return None
    elif n == 0:
        return 1
    else:
        f = 1
        for i in range(2, n + 1):
            f *= i
        # print(f"Fact({n}) = {f}")  # Print directly within the function
        return f

# Example usage:
num = int(input())
print(fact(num))
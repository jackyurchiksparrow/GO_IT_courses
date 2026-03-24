def caching_fibonacci():
    """Creates a function that can calculate the Fibonacci sequence with caching
    
    Returns:
        The recursive `fibonacci` function

    """
    cache = {}

    def fibonacci(n: int) -> int:
        """Recursively evaluates the Fibonacci sequence up to the given position, caches the intermediate result
        Calculates the sequence efficiently by caching the results not to recalculate the same every time
        
        Args:
            n: int
                A number up to which we evaluate the sequence

        Returns:
            The value of the sequence for given `n`

        """
        if n <= 0:
            return 0
        elif n == 1:
            return 1
        elif n in cache:
            return cache[n]
        
        cache[n] = fibonacci(n-1) + fibonacci(n-2)
        return cache[n]

    return fibonacci

if __name__ == "__main__":
    # Отримуємо функцію fibonacci
    fib = caching_fibonacci()

    # Використовуємо функцію fibonacci для обчислення чисел Фібоначчі
    print(fib(10))  # Виведе 55
    print(fib(15))  # Виведе 610

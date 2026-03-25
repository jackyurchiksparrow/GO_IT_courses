from typing import Callable

def generator_numbers(text: str) -> Callable:
    """Splits the text into words (always assumes text is written correctly by dividing word with spaces) and return a generator of real numbers.
    The definition of proper numbers is in `is_real_num`

    Args:
        text: str
            A text to analyze

    Returns:
        A generator of all real numbers found
    """
    def is_real_num(n: str) -> bool:
        """Checks if the number is real and written correctly
        The number is real if:
            - it is separated with spaces (this function takes a whole word as an input)
            - it is correctly written (the task doesn't specify so we consider everything mathematically possible correct, e.g., leading zeroes) via a dot for floats, not a comma

        Args:
            n: int / float / str and everything else that can be parsed into int/float
                A word that represents a number

        Returns:
            True if the number is real
            False otherwise
        
        """
        try:
            float(n)
            return True
        except ValueError:
            return False

    splitted_text = text.split(" ")
    return filter(lambda x: is_real_num(x), splitted_text[1:-1]) # since the first and the last elements can't ever be separated with spaced from both sides -> ignore them


def sum_profit(text: str, func: Callable) -> float:
    """Sums all real numbers in the text
    
    Args:
        text: str
            A text in which to look for numbers and sum
        func: generator
            A generator that filters the text to contain only real numbers

    Returns:
        A sum of all real numbers in the text

    """
    return sum([float(real_num) for real_num in func(text)])


if __name__ == "__main__":
    text = "22 Загальний дохід працівника складається з декількох частин: 1000.01 як основний дохід, доповнений додатковими надходженнями 27.45 і 324.00 доларів. 28"
    #text2 = "fff   0.2   05.55 06 3gwtg wgegweg weg052 25g oogafg"

    total_income = sum_profit(text, generator_numbers)
    print(f"Загальний дохід: {total_income}")
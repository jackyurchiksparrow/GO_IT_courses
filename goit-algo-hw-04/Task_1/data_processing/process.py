from typing import Tuple, List

from validators.validators import expect_numerical_list, expect_number


def mean(numbers: List) -> float:
    """Calculates arithmetic average of numbers given in the list
    
    Args:
        numbers: a non-empty list of numbers
    
    Returns:
        mean(numbers)

    """
    if expect_numerical_list(mean.__name__, numbers, require_non_empty=True):
        return sum(numbers) / len(numbers)


def total_salary(path: str) -> Tuple[float, float]:
    """Reads the .txt file given in 'path', calculates total and average of numbers in it.
    The file is expected to have each line as: name,number
    If specified more than one number per person, only the first is takes into account

    Args:
        path: str
            a string representing the file path including its name (assumed .txt)

    Returns:
        Tuple(total_salary, average_salary) across all staff
    
    """
    
    # if any error emerges, catch them in "except"
    try:
        # open the file, validate, read and close
        with open(path, 'r', encoding="utf-8") as file:
            numbers = []
            file_content = file.readlines()
            for line in file_content:
                if "," not in line:
                    raise ValueError(f"Wrong format in read file \n each line must be: name,number \n got: {line}")

                num = line.strip().split(",")[1]

                if expect_number(total_salary.__name__, num):
                    numbers.append(float(num))

            return (sum(numbers), mean(numbers))

    except FileNotFoundError:
        raise ValueError(f"Sorry, file not found in {path}")
    except Exception as err:
        raise ValueError(f"Error while reading the file: {err}")
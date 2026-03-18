from datetime import datetime
import random
import re

def get_days_from_today(date_str: str) -> int:
    """Calculates the quantity of days between the current and the specified date.
    
    Args:
        date_str: str
            a string representing a date in the YYYY-MM-DD format

    Returns:
        False if date_str is misspecified
        Integer of days between today and the specified date (negative if future dates are given)
    """

    # ----- 1. validate the parameter type -----
    if not isinstance(date_str, str):
        raise ValueError(f" the type of 'date' was expected to be either datetime.date or str, got: {type(date_str)}")

    # ----- 2. confirm the validity of the date and its format -----
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d").date()       # only extract date
        
        return int((datetime.today().date() - date).days)           # parse to int and find the difference of two date objects
    except ValueError as err:                                       # catch and log the error
        print(err)

    return False                                                    # False is the param is wrong


def get_numbers_ticket(min_b: int, max_b: int, quantity: int):
    """Generates `quantity` pieces of unique numbers between `min_b` and `max_b`

    Args:
        min_b: int - minimum boundary
        max_b: int - maximum boundary
        quantity: int - the quantity of generated unique numbers

    Returns:
        [] if parameter(s) misspecified
        a list of unique numbers otherwise
    """
    # define allowed ranges
    allowed_range = range(1,1001)           # range based on the task constraints
    specified_range = range(min_b, max_b)   # range based on the user specifications

    # ----- 1. Validate the input -----
    if min_b not in allowed_range or max_b not in allowed_range or quantity not in allowed_range:
        print(f"min_b, max_b and quantity must be in the {allowed_range}")
        return []

    # Fixed into using length
    if quantity > len(specified_range):
        print(f"You can't expect {quantity} unique tickets in a {specified_range}, can't you?")
        return []

    # ----- 2. Generate an integer each time until `quantity` numbers is generated -----
    tickets = set()

    while len(tickets) < quantity:
        tickets.add(random.randint(min_b, max_b))   # non-unique numbers don't get added

    return sorted(list(tickets))                    # return sorted ticket numbers


def normalize_phone(phone_number: str):
    """Removes all extra syllables from the number and adds "+" if needed
        If the number is Ukrainian (classified with: starts with 380, 80 or 0), ensures the proper Ukrainian format,
        otherwise just remove the extra syllables and adds the "+"

    Args:
        phone_number: str
            a string representing a phone numbers

    Returns:
        same phone number if not a valid Ukrainian number
        normalized phone number into the Ukrainian international format (+380...) no spaces otherwise
    """
    # ----- 1. Check param type -----
    if not isinstance(phone_number, str):
        raise ValueError(f"'phone_number' expected to be 'str', got {type(phone_number)}")

    # ----- 2. Remove white spaces, remove extra characters -----
    phone_number = phone_number.strip()
    phone_number_parsed = re.sub(r"[^\+0-9]", "", phone_number)    # remove everything BUT (^) '+' and numbers

    # ----- 3. FIX: Ukrainian number logic -----
    if phone_number_parsed.startswith('380'):
        phone_number_parsed = '+' + phone_number_parsed
    elif phone_number_parsed.startswith('80'):
        phone_number_parsed = '+3' + phone_number_parsed
    elif phone_number_parsed.startswith('0'):
        phone_number_parsed = '+38' + phone_number_parsed
    elif not phone_number_parsed.startswith('+'):
        phone_number_parsed = '+' + phone_number_parsed
    else:
        # we don't check other countries' numbers based on the task, we only remove the extra characters and add the "+"
        # if we want, we may use the phonenumbers's .parse() and .is_valid_number() if need to and return the status
        pass

    return phone_number_parsed




if __name__ == "__main__":
    # ----- 1. Days between today and the specified date -----
    print("---------- 1st task ----------")
    today = datetime.today().date()
    # test case positive date
    print(f"{today} - 2021-10-09 = {get_days_from_today('2021-10-09')}")
    # test case negative date
    print(f"{today} - 2066-12-11 = {get_days_from_today('2066-12-11')}")
    # test case 0
    print(f"{today} - 2066-12-11 = {get_days_from_today(str(today))}")

    # ----- 2. Days between today and the specified date -----
    # --- FIX + test cases listed in the comment ---
    print("---------- 2nd task ----------")
    print("Ваші лотерейні числа [-10, 10] (5):", get_numbers_ticket(-10,10,5))
    print("Ваші лотерейні числа [1000, 1200] (10):", get_numbers_ticket(1000,1200,10))
    print("Ваші лотерейні числа [10, 4] (5):", get_numbers_ticket(10,4,5))
    print("Ваші лотерейні числа [10, 14] (6):", get_numbers_ticket(10,14,6))

    print("Ваші лотерейні числа [1, 49] (6):", get_numbers_ticket(10,14,6))
    print("Ваші лотерейні числа [5, 10] (4):", get_numbers_ticket(5, 10, 4))

    # ----- 3. Days between today and the specified date -----
    print("---------- 3rd task ----------")

    raw_numbers = [
        "067\\t123 4567",
        "(095) 234-5678\\n",
        "+380 44 123 4567",
        "380501234567",
        "    +38(050)123-32-34",
        "     0503451234",
        "(050)8889900",
        "38050-111-22-22",
        "38050 111 22 11   ",
        "+90 532 456 78 90",
        "90 532 456 78 90",
        "532 456 78 90"
    ]

    print("Нормалізовані номери телефонів для SMS-розсилки:", [normalize_phone(num) for num in raw_numbers])
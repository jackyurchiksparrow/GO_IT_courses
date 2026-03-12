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


def get_number_tickets(min_b: int, max_b: int, quantity: int):
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

    if quantity > max(specified_range):
        print(f"You can't expect {quantity} unique tickets in a {specified_range}, can't you?")
        return []

    # ----- 2. Generate an integer each time until `quantity` numbers is generated -----
    tickets = set()

    while len(tickets) < quantity:
        tickets.add(random.randint(min_b, max_b))   # non-unique numbers don't get added

    return sorted(list(tickets))                    # return sorted ticket numbers


def normalize_phone(phone_number: str):
    """Normalizes a string phone number into the Ukrainian international format (+380...) no spaces
        (assumes "+" can't be in the middle of the string)

    Args:
        phone_number: str
            a string representing a phone numbers

    Returns:
        raises ValueError if the format is misspecified; logs if the number is of wrong length
        normalized phone number into the Ukrainian international format (+380...) no spaces otherwise
    """
    # ----- 1. Check param type -----
    if not isinstance(phone_number, str):
        raise ValueError(f"'phone_number' expected to be 'str', got {type(phone_number)}")

    # ----- 2. Remove white spaces, remove extra characters -----
    phone_number = phone_number.strip()
    phone_number = re.sub(r"[^\+0-9]", "", phone_number)    # remove everything BUT (^) '+' and numbers

    # no '+' in the beginning -> add 
    if phone_number[0] != "+":
        phone_number = "+" + phone_number

    # no '38' after the added '+' -> add
    if phone_number[1:3] != "38": 
        phone_number = phone_number[0] + "38" + phone_number[1:]

    # wrong length -> incorrect number provided
    if len(phone_number) != 13:
        print(f"Attention! The number's format might be incorrect in {phone_number}")

    return phone_number



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
    print("---------- 2nd task ----------")
    print("Ваші лотерейні числа [1, 49] (6):", get_number_tickets(1, 49, 6))
    print("Ваші лотерейні числа [5, 10] (4):", get_number_tickets(5, 10, 4))

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
    ]

    print("Нормалізовані номери телефонів для SMS-розсилки:", [normalize_phone(num) for num in raw_numbers])
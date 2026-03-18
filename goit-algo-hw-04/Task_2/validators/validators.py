def expect_number(func_name: str, func_param, expect_positive: bool = True):
    """Validates the input to be a (positive) number (empty / non-empty). Also allows numbers in strings: "22", '1' etc.'
    
    Args:
        func_name: str
            a name of the parent function from which it is called (for logging)
        func_param: a number
            a number to check
        expect_positive: bool
            if True, expects the positive number
            if False, all numbers are allowed

    Returns:
        True if all checks have passed
    """

    # try to parse to number (checks the string as well)
    try:
        num = int(func_param)

        if expect_positive:
            if num < 0:
                raise ValueError(f"'{func_name}' expected a positive number, received: {num}")

    except ValueError:
        # if not parsed -> incorrect type or empty
        raise ValueError(f"'{func_name}' got an empty/invalid parameter.\n Expected: a positive number OR non-empty string with number, \n got: {func_param} of {type(func_param)}")

    return True
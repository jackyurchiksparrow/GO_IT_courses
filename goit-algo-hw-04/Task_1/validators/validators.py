from typing import List

def expect_numerical_list(func_name: str, func_param: List, require_non_empty: bool = True):
    """Validates the input to be a list of numbers (empty / non-empty). Also allows numbers in strings: [25, 55, "22", "1"]
    
    Args:
        func_name: str
            a name of the parent function from which it is called (for logging)
        func_param: a list
            a list of numbers to check
        require_non_empty: boolean
            if True, expects a non-empty list
            if False, allows an empty list

    Returns:
        True if all checks have passed

    """

    # check if it's a list
    if not isinstance(func_param, List):
        raise ValueError(f"'{func_name}' got invalid param type. Expected: List, got: {type(func_param)}")

    # check if the list isn't empty if require_non_empty == True
    if require_non_empty:
        if len(func_param) == 0:
            raise ValueError(f"'{func_name}' got invalid parameter: an empty list")
        
    # check if all elements in it are numbers or string numbers
    for num in func_param:
        expect_number(func_name, num)
        
    return True

def expect_number(func_name: str, func_param):
    """Validates the input to be a number. Also allows numbers in strings: "22", '1' etc.
    
    Args:
        func_name: str
            a name of the parent function from which it is called (for logging)
        func_param: a number
            a number to check

    Returns:
        True if all checks have passed
    """

    # try to parse to number (checks the string as well)
    try:
        float(func_param)
    except ValueError:
        # if not parsed -> incorrect type or empty
        raise ValueError(f"'{func_name}' got an empty/invalid parameter.\n Expected: number OR non-empty string, \n got: {func_param} of {type(func_param)}")

    return True
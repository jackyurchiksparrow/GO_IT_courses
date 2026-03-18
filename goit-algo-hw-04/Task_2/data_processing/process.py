from typing import List, Dict

from validators.validators import expect_number

def get_cats_info(path: str) -> List[Dict]:
    """Reads the .txt file given in 'path', 
    The file is expected to have each line as: 
        - Doesn't allow non-unique ids
        - Allows empty names
        - Checks the age to be positive or unset (0)

    Args:
        path: str
            a string representing the file path including its name (assumed .txt)

    Returns:
        A list of dictionaries:
        [
            {"id": "...", "name": "...", "age": 0},
            ...
        ]
    
    """
    
    # if any error emerges, catch them in "except"
    try:
        # open the file, validate, read and close
        with open(path, 'r', encoding="utf-8") as file:
            result = []
            unique_ids = set()
            file_content = file.readlines()
            for i, line in enumerate(file_content):
                if "," not in line:
                    raise ValueError(f"Wrong format in read file \n each line must be: name,number \n got: {line}")

                id, name, age = line.strip().split(",")
                curr_obj = {"id": "", "name": "", "age": 0}
                
                if name:
                    curr_obj['name'] = name
                    
                if expect_number(get_cats_info.__name__, age, expect_positive=True):
                    curr_obj['age'] = int(age)

                # if the id isn't empty and unique
                if id and id not in unique_ids:
                    curr_obj['id'] = id
                    unique_ids.add(id)
                    result.append(curr_obj)
                # if the id is not unique
                elif id in unique_ids:
                    raise ValueError(f"The cat's 'id' field is either empty or not unique. File '{path}', line {i+1}, duplicate id: {id}")

            return result

    except FileNotFoundError:
        raise ValueError(f"Sorry, file not found in {path}")
    except Exception as err:
        raise ValueError(f"Error while reading the file: {err}")
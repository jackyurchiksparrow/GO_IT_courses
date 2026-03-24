import sys
from datetime import datetime

# valid levels to expect from logs (for better log file validation)
_LEVELS = ["info", "debug", "error", "warning"]

def parse_log_line(line: str) -> dict:
    """Parses a line from the log file into components and returns them in a dictionary
        - If the log format doesn't match the expected, all extras go into the 'message' component
        - If there are fewer components than needed, the line is skipped
        - Expected date format: Y-m-d
        - Expected time format: H:M:S
    
    Args:
        line: str
            A line from the log file
            
    Returns:
        A dictionary of parsed components if the format is correct
        [] otherwise
        
    """
    # get all the components
    components = line.strip().split(' ')

    # if there are fewer than 4, the format is incorrect -> skip the line
    if len(components) < 4:
        return []
    
    date, time, level, *msg_pieces = components

    # confirm the validity of date and time formats
    try:
        datetime.strptime(date, "%Y-%m-%d")
        datetime.strptime(time, "%H:%M:%S")
    except ValueError:
        print("WRONG date/time format:", date, time)
        return []
    
    # ensure the level exists
    if level.lower() not in _LEVELS:
        print("WRONG level:", level)
        return []
    

    return {
        'date': date,
        'time': time,
        'level': level,
        'message': " ".join(msg_pieces)
    }

def load_logs(file_path: str) -> list:
    """Reads the file given in 'path'. Each line is parsed via `parse_log_line`

    Args:
        file_path: str
            a string representing the file path including its name (assumed .log)

    Returns:
        A list of parsed logs
    
    """
    logs = []
    try:
        # open the file, validate, read and close
        with open(file_path, 'r', encoding="utf-8") as file:
            file_content = file.readlines()
            for line in file_content:
                # if the line was of the right format, it's saved. Otherwise skipped
                parsed_line = parse_log_line(line)
                if parsed_line:
                    logs.append(parsed_line)

        return logs

    except FileNotFoundError:
        raise ValueError(f"Sorry, file not found in {file_path}")
    except Exception as err:
        raise ValueError(f"Error while reading the file: {err}")
    
def filter_logs_by_level(logs: list, level: str) -> list:
    """Filters the logs list to only contain entries of the selected level
    
    Args:
        logs: list
            A list of parsed logs (expected from `load_logs`)
        level: str
            A string representing the level (expected one of _LEVELS)
            
    Returns:
        A list of filtered logs corresponding to the selected level

    """
    return list(filter(lambda log: log['level'].lower() == level.lower(), logs))

def count_logs_by_level(logs: list) -> dict:
    """Counts logs by levels

    Args:
        logs: list
            A list of parsed logs (expected from `load_logs`)

    Returns:
        A dictionary with final counts 
    
    """
    counts_by_level = {}
    
    for log in logs:
        curr_level = log['level']
        if curr_level in counts_by_level:
            counts_by_level[curr_level] += 1
        else:
            counts_by_level[curr_level] = 1

    return counts_by_level

def display_log_counts(counts: dict):
    """Prints the log counts in a table
    
    Args:
        counts: dict
            A dictionary with counted log levels (expected from `count_logs_by_level`)

    """
    print(f"{'Рівень логування':<17} | {'Кількість'}")
    print("-" * 18 + "|" + "-" * 10)
    for level, count in counts.items():
        print(f"{level:<17} | {count}")

if __name__ == "__main__":
    # analyze user's input
    args = sys.argv
    args_len = len(args)

    try:
        # ensure the required path parameter is given
        if args_len == 1:
            raise ValueError(f'You must specify the path. Example: py main.py "<path>"')
        
        path = args[1]
        logs = load_logs(path)
        log_counts = count_logs_by_level(logs)
        display_log_counts(log_counts)

        # if there is one more parameter, then add the additional level prints
        if args_len == 3:
            # if the used-asked level legitimate
            level = args[2].lower()
            if level in _LEVELS:
                print()
                print(f"Деталі логів для рівня '{level.upper()}':")
                results = filter_logs_by_level(logs, level)
                for row in results:
                    print(f"{row['date']} {row['time']} - {row['message']}")
                
            else:
                raise ValueError(f"Error. No such level: '{level}'")
        elif args_len > 3:
            raise ValueError(f"Error. Too many arguments given. Expected: py main.py <'path'> <'level'>: optional")

    # catch unhandled input errors
    except Exception as e:
        raise ValueError(f"Input error. {e}")
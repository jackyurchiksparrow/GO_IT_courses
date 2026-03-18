from colorama import Fore, Style
from pathlib import Path
import sys

def print_all_files_in_dir(path: Path, depth: int = 0):
        """Helper function: recursively prints files and directories in 'path'. Folders are in blue, other files are in green
        
        Args:
            path: Path
                a string with existing path to a directory
            depth: integer
                a number to control current directory depth for output

        Returns:
            No return value, only prints
        """
        
        items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))      # line written by AI: sort to print directories prior to files
        
        # loop through all files in path
        for file in items:
            # if it's a directory
            if file.is_dir():
                # change color to blue and make the indentation 'depth' times
                print(Fore.BLUE + depth * "\t" + file.name + "/")
                # next directory -> increase the depth
                print_all_files_in_dir(file, depth=depth+1)
            else:
                # if it's a file, not a directory, print in green
                print(Fore.GREEN + depth * "\t" + file.name)
                

def visualize_the_file_tree(path: Path):
    """Checks the path and visualizes the file structure in it recursively via 'print_all_files_in_dir'
    
    Args:
        path: Path
            a string with existing path to a directory
    
    Returns:
        No return value, only prints
    """

    # if the path is invalid / doesn't exist
    if not path.exists():
        raise ValueError(f"Path '{path}' doesn't exist. Are you sure you provided the path in quotes: 'path' or \"path\"?")
    # if the path points to a file, not a directory
    elif not path.is_dir():
        raise ValueError(f"You may only specify the path to a directory, not to a file")

    print_all_files_in_dir(path)    # print the files
    print(Style.RESET_ALL)          # reset all colors back to normal

if __name__ == "__main__":
    try:
        # ensure the path parameter is given
        if len(sys.argv) == 1:
            raise ValueError(f'You must specify the path. Example: py main.py "<path>"')
    
        # validate and visualize
        path = Path(str(sys.argv[1]))
        visualize_the_file_tree(path)
    
    # catch unhandled input path errors
    except Exception as e:
        raise ValueError(f"Input path error. {e}")
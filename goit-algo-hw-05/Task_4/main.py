from contacts.contacts_interaction import *

def parse_input(inp: str, contacts: dict):
    """Parses the user's input. Expects it with spaces; not case-sensitive
        Commands:
            - includes "add" -> adds the contact to contacts if it's not there
            - includes "change" -> updates the contact in contacts by name if it's there
            - includes "phone" -> prints the contact from contacts by name if it's there
            - includes "all" -> prints everything from contacts if it's not empty
            - includes "close" or "exit" -> exits the script

    Args:
        inp: str
            a string provided by user
        contacts: dict
            a dictionary with contacts to work with

    Returns:
        str logs
    """
    command, *args = inp.strip().lower().split(" ")

    match command:
        case "hello":
            return "How can I help you?"
        case add if "add" in command:
            return add_contact(args, contacts)
        case change if "change" in command:
            return change_contact(args, contacts)
        case phone if "phone" in command:
            return show_phone(args, contacts)
        case all if "all" in command:
            return show_all(contacts)
        case exit if command in ["close", "exit"]:
            return "Good bye!"
        case _:
            return "Invalid command."


def main():
    """Runs an infinite loop with 'parse_input' continuously asking user for input until they exit

    Args:
        -

    Returns:
        -
    """

    # local instance of contact book
    contacts = {}

    print("Welcome to the assistant bot!")
    while True:
        inp = input("Enter a command: ")
        result = parse_input(inp, contacts)

        print(result)

        if result == "Good bye!":
            break
        elif result == "Show all contacts":
            for name, num in contacts.items():
                print(f"{name} : {num}")


if __name__ == "__main__":
    main()
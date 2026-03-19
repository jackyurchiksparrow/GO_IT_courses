import sys

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
        -
    """
    command, *args = inp.strip().lower().split(" ")

    match command:
        case "hello":
            print("How can I help you?")
        case add if "add" in command:
            if add_contact(args, contacts):
                print("Contact added.")
        case change if "change" in command:
            if change_contact(args, contacts):
                print("Contact updated.")
        case phone if "phone" in command:
            show_phone(args, contacts)
        case all if "all" in command:
            show_all(contacts)
        case exit if command in ["close", "exit"]:
            print("Good bye!")
            sys.exit()
        case _:
            print("Invalid command.")


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
        parse_input(inp, contacts)


if __name__ == "__main__":
    main()
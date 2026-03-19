from typing import List

def add_contact(args: List, contacts: dict):
    """Adds an entry of `name`: `number` from args into the `contacts` dict
        doesn't validate names or phone number format and order (only checks the names to be unique and not present in contacts)
    
    Args:
        args: list
            an ordered list of params to parse, expects 2 - name, number
        contacts: dict
            a dictionary with contacts to add to

    Returns:
        True if a contact was added successfully
        False otherwise
    """

    # if there are not 2 params (name, number) -> notify and return
    if len(args) != 2:
        print(f"Error trying to add contact. Correct syntax: 'add <unique_contact_name> <contact_number>'")
        return False
    
    name, number = args

    # if name is not unique -> notify and return
    if name in contacts:
        print(f"{name} already exists, their contact: {contacts[name]}. Please, use 'change <unique_contact_name> <new_number>' if you'd like to change it.")
        return False

    contacts[name] = number
    return True

def change_contact(args: List, contacts: dict):
    """Changes the entry of `name`: `number` from args in the `contacts` dict
        doesn't validate names or phone number format and order (only checks the names to be unique and present)
    
    Args:
        args: list
            an ordered list of params to parse, expects 2 - name, number
        contacts: dict
            a dictionary with contacts to change in

    Returns:
        True if a contact was updated successfully
        False otherwise
    """

    # if there are not 2 params (name, number) -> notify and return
    if len(args) != 2:
        print(f"Error trying to update contact. Correct syntax: 'change <unique_contact_name> <new_number>'.")
        return False
    
    name, number = args
    
    # if name is not in contacts -> nothing to change, notify and return
    if name not in contacts:
        print(f"Contact '{name}' is't in contacts. Please, use 'add <unique_contact_name> <contact_number>' if you'd like to add them.")
        return False

    contacts[name] = number
    return True

def show_phone(args: List, contacts: dict):
    """Prints the contact entry from 'contacts' by 'name'
        checks for presence of 'name' in contacts
    
    Args:
        args: list
            an ordered list of params to parse, expects 1 - name
        contacts: dict
            a dictionary with contacts to change in

    Returns:
        True if a contact was printed successfully
        False otherwise
    """

    # if there is not 1 param (name) -> notify and return
    if len(args) != 1:
        print(f"Error trying to show contact. Correct syntax: 'phone  <unique_contact_name>'.")
        return False
    
    name = args[0]

    # if the contact is in contacts, print it, otherwise - notify and return
    if name in contacts:
        print(contacts[name])
        return True
    else:
        print(f"Contact '{name}' doesn't exist.")
        return False


def show_all(contacts: dict):
    """Prints all contact entries from 'contacts' if not empty

    Args:
        contacts: dict
            a dictionary with contacts to print from

    Returns:
        True if 'contacts' is not empty
        False otherwise
    """

    if not contacts:
        print("There are no contacts to print")
        return False

    for name, num in contacts.items():
        print(f"{name} : {num}")
    
    return True
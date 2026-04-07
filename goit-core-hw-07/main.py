from collections import UserDict
from datetime import datetime, date, timedelta
from typing import List


def input_error(func):
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as v_err:
            return f"{v_err}"
        except KeyError:
            return "Contact not found."
        except IndexError:
            return "Invalid number of arguments. Please check the command syntax."
        except AttributeError:
            return "Contact does not exist."
        
    return inner


@input_error
def change_contact(args: List, book: 'AddressBook'):
    """Syntax: change <name> <old_number> <new_number>
    
    Args:
        args: list
        book: AddressBook
    """
    
    name, old_number, new_number = args

    person_record = book.find(name)
    if person_record:
        person_record.edit_phone(old_number, new_number)
    else:
        return f"Contact '{name}' isn't in contacts. Please, use 'add <unique_contact_name> <contact_number>' if you'd like to add them."

    
    return "Contact updated."

@input_error
def add_contact(args: List, book: 'AddressBook'):
    """Syntax: add <name> <phone1> <phone2> ...
    If the contact exists, adds unique numbers
    
    Args:
        args: list
            an ordered list of params to parse, expects 3+
        book: AddressBook
    """
    
    name, *numbers = args

    # if name is not unique
    if name in book:
        contacts_added = 0
        person_rec = book.find(name)
        for num in numbers:
            if not person_rec.find_phone(num):
                person_rec.add_phone(num)
                contacts_added += 1
        if contacts_added > 0:
            return "Contact updated"
        return f"The phone numbers already present for '{name}'"

    new_record = Record(name)
    for num in numbers:
        new_record.add_phone(num)

    book.add_record(new_record)
 
    return "Contact added."

@input_error
def show_phone(args: List, book: 'AddressBook'):
    """Prints the contact entry from AddressBook if exists
    Syntax: phone <name>
    
    Args:
        args: list
            an ordered list of params to parse, expects 1
        book: AddressBook
    """
    
    name = args[0]
    
    person_record = book.find(name)
    if not person_record:
        return f"'{name}' is not in contacts"
    
    return person_record

@input_error
def show_all(book: 'AddressBook'):
    """Prints all Record entries from input_error
    Syntax: all

    Args:
        book: AddressBook
    """

    if not book:
        return "There are no contacts to print"

    return "Show all contacts"

@input_error
def add_birthday(args, book: 'AddressBook'):
    """Syntax: add-birthday <name> <birth date: str(DD.MM.YY)>
    Adds a valid birthday date to a contact if record exists

    Args:
        book: AddressBook
        args: list
            an ordered list of params to parse, expects 2

    """
    name, date = args

    person_rec = book.find(name)
    person_rec.add_birthday(date)

    return "Birthday added"

@input_error
def show_birthday(args, book: 'AddressBook'):
    """Syntax: show-birthday <name>
    Shows birthday date of a contact if set

    Args:
        book: AddressBook

    """
    name = args[0]
    return book.find(name).birthday or "Birthday isn't set"

@input_error
def birthdays(book: 'AddressBook'):
    """Syntax: birthdays
    Returns all birthday dates in the near future (7 days)

    Args:
        book: AddressBook

    """
    upcoming_birthdays = book.get_upcoming_birthdays()
    
    if not upcoming_birthdays:
        return "No upcoming birthdays"
    
    return upcoming_birthdays

class Field:
    """Base class for record fields."""
    def __init__(self, field_val):
        self.value = field_val

    # string representation for the class objects for printing
    def __str__(self):
        return str(self.value)
    
class Name(Field):
    """Class for storing a contact name. Mandatory field."""
    def __init__(self, name):
        super().__init__(name) # call the parent constructor to save the attribute value

class Phone(Field):
    """Class for storing a phone number with 10-digit validation."""
    def __init__(self, phone_num: str):
        dig_sum = 0
        for ch in phone_num:
            if ch.isdigit():
                dig_sum += 1
            else:   # we also forbid "+" because 10 digits means no country code
                raise ValueError(f"Phone number may only numbers and exactly 10 of them")
        
        if dig_sum != 10:
            raise ValueError("Phone number must contain exactly 10 digits!")

        super().__init__(phone_num) # call the parent constructor to save the attribute value

class Birthday(Field):
    def __init__(self, value):
        try:
            dt_birthday = datetime.strptime(value, "%d.%m.%Y")
            super().__init__(dt_birthday)
        except ValueError:
            raise ValueError("Invalid date format. Use a STRING with a format of DD.MM.YYYY")


class Record:
    """
    Class representing a contact record. 
    Stores a Name object and a list of Phone objects.
    """
    def __init__(self, name):
        self.name = Name(name)
        self.phones = []
        self.birthday = None

    def add_phone(self, phone_num: str):
        """Adds a new Phone object to the record."""
        self.phones.append(Phone(phone_num))

    def remove_phone(self, phone_num: str):
        """Removes a phone number from the record if it exists."""
        phone_obj = self.find_phone(phone_num)
        
        if phone_obj:
            self.phones.remove(phone_obj)

    def edit_phone(self, old_phone_num: str, new_phone_num : str):
        """
        Replaces an existing phone number with a new one.
        Raises ValueError if the old number is not found.
        """
        phone_obj = self.find_phone(old_phone_num)

        if phone_obj:
            self.add_phone(new_phone_num)
            self.remove_phone(old_phone_num)
        else:
            raise ValueError(f"Number doesn't exist: {old_phone_num}")
    
    def find_phone(self, phone_num: str):
        """Returns a Phone object if it exists in the record, else None."""
        for num in self.phones:
            if num.value == phone_num:
                return num

        return None
    
    def add_birthday(self, birthday_dt: str):
        self.birthday = Birthday(birthday_dt)

    # string representation for the class objects for printing
    def __str__(self):
        return f"Contact name: {self.name.value}, phones: {'; '.join(p.value for p in self.phones)}"

class AddressBook(UserDict):
    """
    Address book container based on UserDict. 
    Stores records using the contact's name as the key.
    """
    def add_record(self, record: Record):
        """Adds a record to the address book. Raises ValueError if name exists."""
        name = record.name.value
        if name in self.data:
            raise ValueError(f"The name {name} already exists!")
        
        self.data[name] = record
        
    # string representation for the class objects for printing
    def __str__(self):
        """Returns a string representation of all records in the book."""
        return "\n".join(str(record) for record in self.data.values())

    def find(self, name: str):
        """Finds a Record object by name. Returns None if not found."""
        return self.data.get(name, None)
    
    def delete(self, name: str):
        """Removes a record from the address book by name."""
        self.data.pop(name, None)

    def get_upcoming_birthdays(self, days=7):
        def _find_next_weekday(start_date, weekday):
            days_ahead = weekday - start_date.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return start_date +  timedelta(days=days_ahead)
        
        def _date_to_string(date):
            return date.strftime("%d.%m.%Y")

        def _adjust_for_weekend(birthday):
            if birthday.weekday() >= 5:
                return _find_next_weekday(birthday, 0)
            return birthday

        upcoming_birthdays = []
        today = date.today()

        for username, usr_data in self.data.items():
            # Skip contacts without a birthday
            if usr_data.birthday is None:
                continue
            
            # Встановлюємо дату народження на поточний рік
            birthday_this_year = usr_data.birthday.value.replace(year=today.year).date()

            # 1. Перевіряємо, чи не минув день народження вже в цьому році
            if birthday_this_year < today:
                birthday_this_year = birthday_this_year.replace(year=today.year + 1)

            # 2. Перевіряємо, чи припадає день народження в інтервал 7 днів
            if 0 <= (birthday_this_year - today).days <= days:
                
                # 3. Переносимо на понеділок, якщо це вихідний
                congratulation_date = _adjust_for_weekend(birthday_this_year)
                
                congratulation_date_str = _date_to_string(congratulation_date)
                upcoming_birthdays.append({
                    "name": username, 
                    "birthday": congratulation_date_str # congratulation date
                })
                
        return upcoming_birthdays

def parse_input(inp: str):
    """Parses the user's input. Expects it with spaces; not case-sensitive

    Args:
        inp: str
            a string provided by user

    """
    command, *args = inp.strip().lower().split(" ")

    return command, *args

def main():
    print("Welcome to the assistant bot!")
    book = AddressBook()

    while True:
        inp = input("Enter a command: ")
        command, *args = parse_input(inp)

        match command:
            case 'hello':
                print("How can I help you?")
            case 'add':
                print(add_contact(args, book))
            case 'change':
                print(change_contact(args, book))
            case 'phone':
                print(show_phone(args, book))
            case 'all':
                print(show_all(book))
            case 'add-birthday':
                print(add_birthday(args, book))
            case 'show-birthday':
                print(show_birthday(args, book))
            case "birthdays":
                upcoming_birthdays = birthdays(book)
                if not isinstance(upcoming_birthdays, str):
                    for rec in upcoming_birthdays:
                        print(f"{rec['name']} : {rec['birthday']}")
                else:
                    print(upcoming_birthdays)
            case 'exit' | 'close':
                print("Good bye!")
                break
            case _:
                print("Invalid command.")


if __name__ == "__main__":
    main()
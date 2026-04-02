from collections import UserDict

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
        if sum(ch.isdigit() for ch in phone_num) != 10: # we don't perform the length check deliberately, the task doesn't specify
            raise ValueError("Phone number must contain exactly 10 digits!")

        super().__init__(phone_num) # call the parent constructor to save the attribute value

class Record:
    """
    Class representing a contact record. 
    Stores a Name object and a list of Phone objects.
    """
    def __init__(self, name):
        self.name = Name(name)
        self.phones = []

    def add_phone(self, phone_num: str):
        """Adds a new Phone object to the record."""
        self.phones.append(Phone(phone_num))

    def remove_phone(self, phone_num: str):
        """Removes a phone number from the record if it exists."""
        self.phones = [p for p in self.phones if p.value != phone_num]

    def edit_phone(self, old_phone_num: str, new_phone_num : str):
        """
        Replaces an existing phone number with a new one.
        Raises ValueError if the old number is not found.
        """
        for i, num in enumerate(self.phones):
            if num.value == old_phone_num:
                self.phones[i] = Phone(new_phone_num)
                return True
        
        raise ValueError(f"Number doesn't exist: {old_phone_num}")
    
    def find_phone(self, phone_num: str):
        """Returns a Phone object if it exists in the record, else None."""
        for num in self.phones:
            if num.value == phone_num:
                return num

        return None

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
        if name not in self.data:
            return None
        
        return self.data[name]
    
    def delete(self, name: str):
        """Removes a record from the address book by name."""
        self.data.pop(name, None)


if __name__ == "__main__":
    # Створення нової адресної книги
    book = AddressBook()

    # Створення запису для John
    john_record = Record("John")
    john_record.add_phone("1234567890")
    john_record.add_phone("5555555555")

    # Додавання запису John до адресної книги
    book.add_record(john_record)

    # Створення та додавання нового запису для Jane
    jane_record = Record("Jane")
    jane_record.add_phone("9876543210")
    book.add_record(jane_record)

    # Виведення всіх записів у книзі
    print(book)

    # Знаходження та редагування телефону для John
    john = book.find("John")
    john.edit_phone("1234567890", "1112223333")
    
    print(john)

    # Пошук конкретного телефону у записі John
    found_phone = john.find_phone("5555555555")
    print(f"{john.name}: {found_phone}")

    # Видалення запису Jane
    book.delete("Jane")

    print(book)
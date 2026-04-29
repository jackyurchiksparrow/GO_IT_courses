from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure
from pymongo.server_api import ServerApi

REQUESTS_TIMEOUT = 5  # Seconds to wait for a server response


def connect_to_db_atlas(uri: str) -> MongoClient | None:
    """
    Establishes a connection to MongoDB Atlas.

    Uses ServerApi V1 to ensure long-term stability with the cloud cluster.
    """
    client = MongoClient(uri, server_api=ServerApi("1"), serverSelectionTimeoutMS=REQUESTS_TIMEOUT * 1000)

    try:
        # Verify connection by sending a ping command
        client.admin.command("ping")
        print("✅ Successfully connected to MongoDB Atlas!")
    except ConnectionFailure:
        print("❌ Could not connect to MongoDB Atlas. Are the cluster created and credentials ok?")
    else:
        # if no exceptions raised
        return client

    return None


# just for practice, irrelevant to the assignment
def connect_to_db_local(port="27017"):
    """Establishes a connection to a local MongoDB instance (usually in Docker)."""
    # Points to the local Docker container
    client = MongoClient(f"mongodb://localhost:{port}/", serverSelectionTimeoutMS=5000)
    # Send a ping to confirm a successful connection
    try:
        client.admin.command("ping")
        print("✅ Successfully connected to local MongoDB in Docker!")
    except ConnectionFailure:
        print("❌ Could not connect to MongoDB. Is the Docker container running?")
    else:
        # if no exceptions raised
        return client

    return None


def populate_db(db: Database, collection_name: str) -> None:
    """Initializes the database with a starter set of cat documents."""
    cats_data = [
        {"name": "Murzik", "age": 1, "features": ["ходить в капці", "дає себе гладити", "чорний"]},
        {"name": "barsik", "age": 3, "features": ["ходить в капці", "дає себе гладити", "рудий"]},
        {"name": "Lama", "age": 2, "features": ["ходить в лоток", "не дає себе гладити", "сірий"]},
        {"name": "Liza", "age": 4, "features": ["ходить в лоток", "дає себе гладити", "білий"]},
    ]
    inserted = db[collection_name].insert_many(cats_data)

    # Check how many were inserted
    print(f"Inserted {len(inserted.inserted_ids)} cats: {inserted.inserted_ids}")


def read_all(db: Database) -> None:
    """Prints every document currently stored in the 'cats' collection."""
    cursor = db.cats.find({})  # returns pymongo.cursor

    for document in cursor:
        print(document)


def find_one_by_name(db: Database, name: str) -> dict | None:
    """Helper function to find a specific cat dictionary using a case-insensitive regex."""
    # 'i' to make it case-insensitive
    return db.cats.find_one({"name": {"$regex": f"^{name}$", "$options": "i"}})


def read_by_name(db: Database, name: str) -> None:
    """Fetches and displays information for a cat by its name."""
    found = find_one_by_name(db, name)
    if found:
        print(f"✅ Cat '{name}' found: {found}")
    else:
        print(f"❌ No cat found with the name '{name}'.")


def update_age_by_name(db: Database, name: str, new_age) -> None:
    """
    Updates the 'age' field of a cat document.

    Uses $set to change only the age without affecting other fields.
    """
    updated = db.cats.update_one(
        {"name": {"$regex": f"^{name}$", "$options": "i"}},
        {"$set": {"age": new_age}},
    )

    if updated.matched_count > 0:
        if updated.modified_count > 0:
            print(f"✅ Successfully updated age for '{name}' to '{new_age}'.")
            read_by_name(db, name)
        else:
            print(f"Cat '{name}' found, but age was already {new_age}. No changes made.")
    else:
        print(f"❌ No cat found with the name '{name}'.")


def update_append_feature(db: Database, name: str, new_feature: str) -> None:
    """
    Adds a new unique feature to a cat's feature list.

    Uses $addToSet to ensure the feature is only added if it doesn't already exist.
    """
    updated = db.cats.update_one(
        {"name": {"$regex": f"^{name}$", "$options": "i"}},
        {"$addToSet": {"features": new_feature}},
    )

    if updated.matched_count > 0:
        if updated.modified_count > 0:
            print(f"✅ Successfully updated features for '{name}' with '{new_feature}'.")
            read_by_name(db, name)
        else:
            print(f"Cat '{name}' found, but features already had {new_feature}. No changes made.")
    else:
        print(f"❌ No cat found with the name '{name}'.")


def delete_by_name(db: Database, name: str) -> None:
    """Removes a single cat document from the collection by name."""
    deleted = db.cats.delete_one({"name": {"$regex": f"^{name}$", "$options": "i"}})

    # result.deleted_count will be 1 if a document was removed, 0 if not
    if deleted.deleted_count > 0:
        print(f"✅ Successfully deleted cat '{name}'.")
    else:
        print(f"❌ No cat found with the name '{name}', so nothing was deleted.")


def delete_collection(db: Database, collection_name: str) -> None:
    """Completely removes a collection. Used to reset the state for the assignment."""
    if collection_name in db.list_collection_names():
        db[collection_name].drop()
        print(f"✅ Collection '{collection_name}' has been deleted.")


def main():
    """Main execution flow."""
    client = connect_to_db_atlas("mongodb+srv://uradzbq_db_user:111@goit-learn-mongodb.1dyektb.mongodb.net/?appName=goit-learn-mongodb")

    if client:
        # Connect to 'cats' database
        db = client.cats

        # Реалізуйте функцію для видалення всіх записів із колекції.
        delete_collection(db, "cats")

        # Populate
        populate_db(db, "cats")

        # Реалізуйте функцію для виведення всіх записів із колекції.
        print("✅ All cats listed:")
        read_all(db)

        # Реалізуйте функцію, яка дозволяє користувачеві ввести ім'я кота та виводить інформацію про цього кота.
        read_by_name(db, "murzik")

        # Створіть функцію, яка дозволяє користувачеві оновити вік кота за ім'ям.
        update_age_by_name(db, "murzik", 11)

        # Створіть функцію, яка дозволяє додати нову характеристику до списку features кота за ім'ям.
        update_append_feature(db, "murzik", "new_feature")

        # Реалізуйте функцію для видалення запису з колекції за ім'ям тварини.
        delete_by_name(db, "murzik")
    else:
        print("❌ Unsuccessful database connection. Exited.")
        return


if __name__ == "__main__":
    main()

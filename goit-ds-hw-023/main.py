from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure


def connect_to_db(port="27017"):
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


def populate_db(db: Database) -> None:
    inserted = db.cats.insert_many(
        [
            {
                "name": "Murzik",
                "age": 1,
                "features": ["ходить в капці", "дає себе гладити", "чорний"],
            },
            {
                "name": "barsik",
                "age": 3,
                "features": ["ходить в капці", "дає себе гладити", "рудий"],
            },
            {
                "name": "Lama",
                "age": 2,
                "features": ["ходить в лоток", "не дає себе гладити", "сірий"],
            },
            {
                "name": "Liza",
                "age": 4,
                "features": ["ходить в лоток", "дає себе гладити", "білий"],
            },
        ],
    )

    # Check how many were inserted
    print(f"Inserted {len(inserted.inserted_ids)} cats: {inserted.inserted_ids}")


def read_all(db: Database) -> None:
    cursor = db.cats.find({})  # returns pymongo.cursor

    for document in cursor:
        print(document)


def find_one_by_name(db: Database, name: str) -> dict | None:
    # 'i' to make it case-insensitive
    return db.cats.find_one({"name": {"$regex": f"^{name}$", "$options": "i"}})


def read_by_name(db: Database, name: str) -> None:
    found = find_one_by_name(db, name)
    if found.matched_count > 0:
        print(f"✅ Cat '{name}' found: {found}")
    else:
        print(f"❌ No cat found with the name '{name}'.")


def update_age_by_name(db: Database, name: str, new_age) -> None:
    updated = db.cats.update_one(
        {"name": {"$regex": f"^{name}$", "$options": "i"}},
        {"$set": {"age": new_age}},
    )

    if updated.matched_count > 0:
        if updated.modified_count > 0:
            print(f"✅ Successfully updated age for '{name}'.")
            read_by_name(db, name)
        else:
            print(
                f"Cat '{name}' found, but age was already {new_age}. No changes made."
            )
    else:
        print(f"❌ No cat found with the name '{name}'.")


def update_append_feature(db: Database, name: str, new_feature: str) -> None:
    updated = db.cats.update_one(
        {"name": {"$regex": f"^{name}$", "$options": "i"}},
        {"$addToSet": {"features": new_feature}},
    )

    if updated.matched_count > 0:
        if updated.modified_count > 0:
            print(
                f"✅ Successfully updated features for '{name}' with '{new_feature}'."
            )
            read_by_name(db, name)
        else:
            print(
                f"Cat '{name}' found, but features already had {new_feature}. No changes made."
            )
    else:
        print(f"❌ No cat found with the name '{name}'.")


def delete_by_name(db: Database, name: str) -> None:
    deleted = db.cats.delete_one({"name": {"$regex": f"^{name}$", "$options": "i"}})

    # result.deleted_count will be 1 if a document was removed, 0 if not
    if deleted.deleted_count > 0:
        print(f"✅ Successfully deleted cat '{name}'.")
    else:
        print(f"❌ No cat found with the name '{name}', so nothing was deleted.")


def delete_collection(db: Database, collection_name: str) -> None:
    if collection_name in db.list_collection_names():
        db[collection_name].drop()
        print(f"✅ Collection '{collection_name}' has been deleted.")
    else:
        print(f"❌ Collection '{collection_name}' does not exist.")


def main():
    client = connect_to_db()
    db = client.cats
    populate_db(db)

    # Реалізуйте функцію для виведення всіх записів із колекції.
    print("✅ All cats listed:")
    read_all(db)

    # Реалізуйте функцію, яка дозволяє користувачеві ввести ім'я кота та виводить інформацію про цього кота.
    read_by_name(db, "murzik")

    # Створіть функцію, яка дозволяє користувачеві оновити вік кота за ім'ям.
    update_age_by_name(db, "murzik", 11)

    # Створіть функцію, яка дозволяє додати нову характеристику до списку features кота за ім'ям.
    print(update_append_feature(db, "murzik", "new_feature"))

    # Реалізуйте функцію для видалення запису з колекції за ім'ям тварини.
    delete_by_name(db, "murzik")

    # Реалізуйте функцію для видалення всіх записів із колекції.
    delete_collection(db, "cats")


if __name__ == "__main__":
    main()

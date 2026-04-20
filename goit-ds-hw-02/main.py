from create_task_db import create_db
from fill_data_db import insert_data_to_db
from seed import generate_fake_data

MAX_USERS = 30
MAX_TASKS_PER_USER = 5
STATUSES = [("new",), ("in progress",), ("completed",)]


def main():
    """Orchestrates the full workflow: schema creation, data generation, and DB seeding."""
    create_db()

    # Generate and preview sample data
    fake_users, fake_tasks = generate_fake_data(MAX_USERS, MAX_TASKS_PER_USER)

    print(fake_users[:5])
    print()
    print(fake_tasks[:10])

    # Populate the database
    insert_data_to_db(fake_users, fake_tasks)


if __name__ == "__main__":
    main()

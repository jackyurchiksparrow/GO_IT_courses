from create_task_db import create_db
from fill_data_db import insert_data_to_db
from seed import generate_fake_data

MAX_USERS = 30
MAX_TASKS_PER_USER = 5
STATUSES = [("new",), ("in progress",), ("completed",)]


def main():
    create_db()

    fake_users, fake_tasks = generate_fake_data(MAX_USERS, MAX_TASKS_PER_USER)

    print(fake_users[:5])
    print()
    print(fake_tasks[:10])

    insert_data_to_db(fake_users, fake_tasks)


if __name__ == "__main__":
    main()

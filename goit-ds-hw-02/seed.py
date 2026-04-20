import random

import faker

FakeData = tuple[list[tuple], list[tuple]]


def generate_fake_data(users_qty: int, tasks_per_user: int) -> FakeData:
    """
    Generates synthetic data for users and tasks.

    Args:
        users_qty: int
            max quantity of users to create
        tasks_per_user: int
            max quantity of tasks per user to create

    Returns two lists of tuples ready for database insertion.

    """
    fake_data = faker.Faker()

    # Generate user data: (id, fullname, email)
    fake_users = []
    for user_id in range(1, users_qty + 1):
        fake_users.append((user_id, fake_data.name(), fake_data.email()))

    # Generate task data: (title, description, status_id, user_id)
    fake_tasks = []
    for user_id in range(1, users_qty + 1):
        # Each user gets a random number of tasks (0..tasks_per_user)
        for _ in range(random.randint(0, tasks_per_user)):
            title = fake_data.catch_phrase()
            # Construct a pseudo-realistic description
            description = (
                f"{fake_data.bs().capitalize()} to ensure {fake_data.word()} quality."
            )

            # Randomly assign one of the 3 predefined statuses
            # assuming 3 statuses: 'new', 'in progress', 'completed'
            status_id = random.randint(1, 3)
            fake_tasks.append((title, description, status_id, user_id))

    return fake_users, fake_tasks

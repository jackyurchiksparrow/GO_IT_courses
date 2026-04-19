import random

import faker


def generate_fake_data(users_qty, tasks_per_user):
    fake_data = faker.Faker()

    fake_users = []
    for user_id in range(1, users_qty + 1):
        fake_users.append((user_id, fake_data.name(), fake_data.email()))

    fake_tasks = []
    for user_id in range(1, users_qty + 1):
        # Each user gets a random number of tasks
        for _ in range(random.randint(0, tasks_per_user)):
            title = fake_data.catch_phrase()
            description = (
                f"{fake_data.bs().capitalize()} to ensure {fake_data.word()} quality."
            )

            # assuming 3 statuses: new, in progress, completed
            status_id = random.randint(1, 3)

            fake_tasks.append((title, description, status_id, user_id))

    return fake_users, fake_tasks

import sqlite3


def insert_data_to_db(users: list[tuple], tasks: list[tuple]) -> None:
    """
    Connects to the SQLite database and populates tables with provided data.

    Args:
        users: list[tuple]
            ready for insertion list of tuples[(id, name, email),...]
        tasks: list[tuple]
            ready for insertion list of tuples[(title, description, status_id, user_id),...]

    """
    with sqlite3.connect("create_task_db.db") as con:
        cur = con.cursor()

        # Initialize fixed statuses
        sql_to_status = """INSERT INTO status(name) VALUES ("new"), ("in progress"), ("completed");"""
        cur.execute(sql_to_status)

        # Batch insert users
        sql_to_users = """INSERT INTO users(id, fullname, email) VALUES (?, ?, ?);"""
        cur.executemany(sql_to_users, users)

        # Batch insert tasks
        sql_to_tasks = """INSERT INTO tasks(title, description, status_id, user_id) VALUES(?, ?, ?, ?);"""
        cur.executemany(sql_to_tasks, tasks)

        con.commit()

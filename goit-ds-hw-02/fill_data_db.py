import sqlite3


def insert_data_to_db(users, tasks) -> None:
    with sqlite3.connect("create_task_db.db") as con:
        cur = con.cursor()

        sql_to_status = """INSERT INTO status(name) VALUES ("new"), ("in progress"), ("completed");"""
        cur.execute(sql_to_status)

        sql_to_users = """INSERT INTO users(id, fullname, email) VALUES (?, ?, ?);"""
        cur.executemany(sql_to_users, users)

        sql_to_tasks = """INSERT INTO tasks(title, description, status_id, user_id) VALUES(?, ?, ?, ?);"""
        cur.executemany(sql_to_tasks, tasks)

        con.commit()

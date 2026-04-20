import sqlite3


def create_db() -> None:
    """Initializes the database schema by executing the provided .sql script."""
    # read the .sql script
    with open("create_task_db.sql", "r", encoding="utf-8") as f:
        sql = f.read()

    # create db connection (will be created if absent)
    with sqlite3.connect("create_task_db.db") as con:
        cur = con.cursor()
        # execute the script
        cur.executescript(sql)


if __name__ == "__main__":
    create_db()

# Module 2: Введення в SQL бази даних


## File structure

- create_task_db.db - created SQLite database for users and their tasks
- create_task_db.py - python script to create the `create_task_db.db` database if not exists
- create_task_db.sql - the SQL script used by `create_task_db.py ` to create `create_task_db.db`
- seed.py - generates synthetic data for users and tasks
- fill_data_db.py - insert data functionality generated with `seed.py`
- main.py - orchestrates the full workflow: schema creation, data generation, and DB seeding
- poetry.toml, poetry.lock - Poetry virtual environment
- queries.sql - queries (task 2)

---
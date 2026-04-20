PRAGMA foreign_keys = ON; -- sets CASCADE command to work, it is disabled by default

-- Table: users
DROP TABLE IF EXISTS users; -- just a safeguard
CREATE TABLE users (
	id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
	fullname VARCHAR(100) NOT NULL,
	email VARCHAR(100) NOT NULL,
	CONSTRAINT email_uq UNIQUE (email)	-- we could just use 'email VARCHAR(100) NOT NULL UNIQUE', 
										-- but using constraints is the best practice for clarity, convenient change using ALTER
	                                 	-- and if the error occurs, it will refer to this specific 'email_uq' key that we can assign thanks to 'CONSTRAINT'
	                           			-- + you can do combinations of constraints: 'UNIQUE (fullname, email)' not just for email
);

-- Table: status
DROP TABLE IF EXISTS status;
CREATE TABLE status (
	id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
	name VARCHAR(50) NOT NULL,
	CONSTRAINT name_uq UNIQUE (name)
);

-- Table: tasks
DROP TABLE IF EXISTS tasks;
CREATE TABLE tasks (
	id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
	title VARCHAR(100) NOT NULL,
	description TEXT NOT NULL,
	status_id INTEGER NOT NULL,
	user_id INTEGER NOT NULL,
	CONSTRAINT status_id_fk FOREIGN KEY (status_id) REFERENCES status (id)
		ON DELETE CASCADE  -- means if the original primary key is deleted (user), then all related things it was referring to will be deleted
		                   -- if we wanted to just set them to null instead of deleting when the user is gone, the command would be 'ON DELETE SET NULL'
		ON UPDATE CASCADE, -- means if the original primary key is changed (user), then all ids that references it will also be updated
		                   -- they say it's extremely rare, but I find it very useful and would use it always. In fact, I believe it must be made default
	CONSTRAINT user_id_fk FOREIGN KEY (user_id) REFERENCES users (id)
		ON DELETE CASCADE
		ON UPDATE CASCADE
);

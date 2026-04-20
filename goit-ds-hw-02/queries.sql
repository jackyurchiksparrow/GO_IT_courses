--Отримати всі завдання певного користувача. Використайте SELECT для отримання завдань конкретного користувача за його user_id.
SELECT title, description FROM tasks WHERE user_id = 5;
--Вибрати завдання за певним статусом. Використайте підзапит для вибору завдань з конкретним статусом, наприклад, 'new'.
SELECT title, description FROM tasks 
WHERE status_id IN (SELECT id FROM status WHERE id = 1);
--Оновити статус конкретного завдання. Змініть статус конкретного завдання на 'in progress' або інший статус.
UPDATE tasks SET status_id = 2 where id = 2;
--Отримати список користувачів, які не мають жодного завдання. Використайте комбінацію SELECT, WHERE NOT IN і підзапит.
SELECT * FROM users WHERE id NOT IN (SELECT user_id FROM tasks);
--Додати нове завдання для конкретного користувача. Використайте INSERT для додавання нового завдання.
INSERT INTO 
	tasks(title, description, status_id, user_id) 
	VALUES("new manual task", "new manual task description", 1, 3);
--Отримати всі завдання, які ще не завершено. Виберіть завдання, чий статус не є 'завершено'.
SELECT * FROM tasks WHERE status_id != 3;
--Видалити конкретне завдання. Використайте DELETE для видалення завдання за його id.
DELETE FROM tasks WHERE id = 70;
--Знайти користувачів з певною електронною поштою. Використайте SELECT із умовою LIKE для фільтрації за електронною поштою.
SELECT * FROM users where email LIKE '%.net';
--Оновити ім'я користувача. Змініть ім'я користувача за допомогою UPDATE.
UPDATE users SET fullname = "John Doe" WHERE id = 1;
--Отримати кількість завдань для кожного статусу. Використайте SELECT, COUNT, GROUP BY для групування завдань за статусами.
SELECT status_id, COUNT(id) AS count FROM tasks GROUP BY status_id;
--Отримати завдання, які призначені користувачам з певною доменною частиною електронної пошти. Використайте SELECT з умовою LIKE в поєднанні з JOIN, щоб вибрати завдання, призначені користувачам, чия електронна пошта містить певний домен (наприклад, '%@example.com').
SELECT 
	tasks.id as task_id, fullname, title, description, status_id 
FROM tasks INNER JOIN users ON tasks.user_id = users.id 
WHERE users.email LIKE '%.net';
--Отримати список завдань, що не мають опису. Виберіть завдання, у яких відсутній опис.
SELECT * FROM tasks WHERE description = '';
--Вибрати користувачів та їхні завдання, які є у статусі 'in progress'. Використайте INNER JOIN для отримання списку користувачів та їхніх завдань із певним статусом.
SELECT 
	users.id as user_id, fullname, tasks.id as task_id, title, description 
FROM users 
INNER JOIN tasks ON users.id = tasks.user_id 
WHERE tasks.status_id = 2;
--Отримати користувачів та кількість їхніх завдань. Використайте LEFT JOIN та GROUP BY для вибору користувачів та підрахунку їхніх завдань.
SELECT 
	users.fullname, users.id as user_id, COUNT(tasks.user_id) as task_count 
FROM users 
LEFT JOIN tasks ON users.id = tasks.user_id 
GROUP BY users.id ORDER BY task_count;

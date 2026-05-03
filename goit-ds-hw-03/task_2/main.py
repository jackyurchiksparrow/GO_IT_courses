import json

import requests
from bs4 import BeautifulSoup, Tag
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure
from pymongo.server_api import ServerApi

# Configuration constants
STATUS_CODE_OK = 200
REQUESTS_TIMEOUT = 5  # Seconds to wait for a server response


def handle_file_errors(func):
    """Decorator to centralize error handling for file I/O operations."""

    def wrapper(filename, *args, **kwargs):
        try:
            return func(filename, *args, **kwargs)
        except FileNotFoundError as err:
            msg = f"'{filename}' not found."
            raise FileNotFoundError(msg) from err
        except json.JSONDecodeError as err:
            msg = f"'{filename}' has invalid JSON."
            raise ValueError(msg) from err
        except PermissionError as err:
            msg = f"Permission denied for '{filename}'."
            raise PermissionError(msg) from err
        except Exception as err:
            msg = f"Unexpected error with {filename}: {err}"
            raise RuntimeError(msg) from err

    return wrapper


@handle_file_errors
def load_from_json(filename: str) -> list | dict:
    """Loads and parses data from a JSON file."""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


@handle_file_errors
def save_to_json(filename: str, data: list | dict) -> None:
    """Serializes data to a JSON file with proper UTF-8 encoding and indentation."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def connect_to_db_atlas(uri: str) -> MongoClient | None:
    """
    Establishes a connection to MongoDB Atlas.

    Uses ServerApi V1 to ensure long-term stability with the cloud cluster.
    """
    client = MongoClient(uri, server_api=ServerApi("1"), serverSelectionTimeoutMS=REQUESTS_TIMEOUT * 1000)

    try:
        # Verify connection by sending a ping command
        client.admin.command("ping")
        print("✅ Successfully connected to MongoDB Atlas!")
    except ConnectionFailure:
        print("❌ Could not connect to MongoDB Atlas. Are the cluster created and credentials ok?")
    else:
        # if no exceptions raised
        return client

    return None


def save_data_to_atlas(db: Database, collection_name: str, data: list[dict]) -> None:
    """Inserts a list of dictionaries into a specified MongoDB collection."""
    inserted = db[collection_name].insert_many(data)

    # Check how many were inserted
    print(f"✅ Inserted {len(inserted.inserted_ids)} entries: {inserted.inserted_ids[:10]}")


def delete_collection(db: Database, collection_name: str) -> None:
    """Drops a collection from the database to ensure a clean state for re-runs."""
    if collection_name in db.list_collection_names():
        db[collection_name].drop()
        print(f"✅ Collection '{collection_name}' has been deleted.")


def parse_author(quote: Tag) -> dict:
    """
    Scrapes detailed author information from their 'about' page.

    Triggered only when a new author is encountered during main scraping.
    """
    authors_block = quote.select("span")[1]
    about_url = authors_block.select_one("a")["href"]

    html_doc = requests.get(f"https://quotes.toscrape.com/{about_url}", timeout=REQUESTS_TIMEOUT)
    if html_doc.status_code == STATUS_CODE_OK:
        soup = BeautifulSoup(html_doc.content, "html.parser")
        name = soup.select_one("h3.author-title").get_text().strip()
        born_date = soup.select_one("span.author-born-date").get_text().strip()
        born_location = soup.select_one("span.author-born-location").get_text().strip()
        description = soup.select_one(".author-description").get_text().strip()
    else:
        print(f"Wasn't able to get https://quotes.toscrape.com/{about_url}. Status code: {html_doc.status_code}")

    return {"fullname": name, "born_date": born_date, "born_location": born_location, "description": description}


def start_scraping() -> tuple[list[dict], list[dict]]:
    """
    Iterates through all pages of quotes.toscrape.com.

    Collects quotes, tags, and detailed author bios.
    """
    all_quotes_data = []
    all_authors_data = {}  # dict to avoid duplicates
    page = 1

    while True:
        # Fetch the page
        response = requests.get(f"https://quotes.toscrape.com/page/{page}", timeout=REQUESTS_TIMEOUT)
        soup = BeautifulSoup(response.content, "html.parser")
        quotes = soup.select(".quote")

        # Break loop if no more quotes are found or server error occurs
        if response.status_code != STATUS_CODE_OK or not quotes:
            break

        print(f"Scraping quotes... (page #{page})")

        for quote in quotes:
            tag_elements = quote.select("div.tags a")
            tags_list = [tag.get_text() for tag in tag_elements]
            author = quote.select_one("small.author").get_text()
            text = quote.select_one("span.text").get_text() if quote.select_one(".text") else ""

            all_quotes_data.append(
                {
                    "tags": tags_list,
                    "author": author,
                    "quote": text,
                },
            )

            # add unique author info (uniqueness judged by name)
            if author not in all_authors_data:
                all_authors_data[author] = parse_author(quote)

        page += 1

    print("✅ Done scraping")
    return all_quotes_data, list(all_authors_data.values())


def main():
    """Orchestrates the scraping, local storage, and database upload processes."""
    client = connect_to_db_atlas("mongodb+srv://uradzbq_db_user:111@goit-learn-mongodb.1dyektb.mongodb.net/?appName=goit-learn-mongodb")

    if client:
        # delete collections if exist
        delete_collection(client.db_quotes, "quotes")
        delete_collection(client.db_authors, "authors")

        # create the collections
        db_quotes = client.db_quotes
        db_authors = client.db_authors

        # start scraping quotes and authors
        all_quotes_data, all_authors_data = start_scraping()

        # save the scraped to the json
        save_to_json("quotes.json", all_quotes_data)
        save_to_json("authors.json", all_authors_data)

        # load back from the json
        quotes_to_upload = load_from_json("quotes.json")
        authors_to_upload = load_from_json("authors.json")

        # save json-loaded data to Atlas
        save_data_to_atlas(db_quotes, "quotes", quotes_to_upload)
        save_data_to_atlas(db_authors, "authors", authors_to_upload)
    else:
        print("❌ Unsuccessful database connection. Exited.")
        return


if __name__ == "__main__":
    main()

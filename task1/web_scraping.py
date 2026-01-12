import requests
from bs4 import BeautifulSoup
import pandas as pd

# Base URL du site
BASE_URL = "https://quotes.toscrape.com"

# Liste pour stocker les données
quotes_data = []

page = 1

while True:
    url = f"{BASE_URL}/page/{page}/"
    response = requests.get(url)

    if response.status_code != 200:
        break

    soup = BeautifulSoup(response.text, "html.parser")
    quotes = soup.find_all("div", class_="quote")

    if not quotes:
        break

    for quote in quotes:
        text = quote.find("span", class_="text").get_text()
        author = quote.find("small", class_="author").get_text()
        tags = [tag.get_text() for tag in quote.find_all("a", class_="tag")]

        quotes_data.append({
            "quote": text,
            "author": author,
            "tags": ", ".join(tags)
        })

    print(f"Page {page} scraped successfully")
    page += 1

# Conversion en DataFrame
df = pd.DataFrame(quotes_data)

# Sauvegarde en CSV
df.to_csv("scraped_quotes.csv", index=False, encoding="utf-8")

print("Web scraping completed. Data saved to scraped_quotes.csv")

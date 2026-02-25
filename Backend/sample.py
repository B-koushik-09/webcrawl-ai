import json

with open("data/scraped_pages.json", "r", encoding="utf-8") as f:
    data = json.load(f)

pages = data["pages"]   # this is a LIST

with open("pages_list.txt", "w", encoding="utf-8") as out:
    for i, page in enumerate(pages, 1):
        url = page.get("url", "")
        title = page.get("title", "")
        text = page.get("text", "")

        out.write(f"{i}. {url}\n")
        out.write(f"   Title: {title}\n")
        out.write(f"   Text length: {len(text)}\n\n")

print("pages_list.txt created successfully!")

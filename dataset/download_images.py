import os
import json
import requests
from urllib.parse import urlsplit

def download_image(url, path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Image saved to {path}")
    else:
        print(f"Failed to download image from {url}")


with open('image_urls.json', 'r') as file:
    image_urls = json.load(file)

main_dir = "data"

if not os.path.exists(main_dir):
    os.makedirs(main_dir)

for key, url in image_urls.items():
    sub_dir = os.path.join(main_dir, key)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    image_path = os.path.join(sub_dir, f"{key}.jpg")
    download_image(url, image_path)

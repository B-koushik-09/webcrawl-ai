import requests
import time

url = "http://127.0.0.1:5000/api/admin/rebuild"
print(f"Triggering rebuild at {url}...")

try:
    response = requests.post(url)
    print(f"Status: {response.status_code}")
    print("Response:", response.json())
except Exception as e:
    print(f"Failed to trigger rebuild: {e}")

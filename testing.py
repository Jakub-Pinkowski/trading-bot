import requests

url = "https://localhost:5000/v1/api"
headers = {
    "Host": "api.ibkr.com",
    "User-Agent": "Python Requests",
    "Accept": "*/*",
    "Connection": "keep-alive"
}

print(f"Request URL: {url}")

try:
    response = requests.get(url, headers=headers, verify=False)
    print(f"Status Code: {response.status_code}")
    print(f"Response Content: {response.text}")
except requests.exceptions.ConnectionError as e:
    print(f"Connection Error: {e}")
except requests.exceptions.RequestException as e:
    print(f"Request Exception: {e}")
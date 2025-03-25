import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def confirm_status():
    base_url = "https://localhost:5001/v1/api/"
    endpoint = "iserver/auth/status"

    auth_req = requests.get(url=base_url + endpoint, verify=False)
    print(auth_req)
    print(auth_req.text)


if __name__ == "__main__":
    print("Testing")
    confirm_status()

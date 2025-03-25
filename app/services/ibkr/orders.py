import requests


def place_order(conid, order):
    account_id = "DUE343675"

    base_url = "https://localhost:5001/v1/api/"
    endpoint = f"iserver/account/{account_id}/orders"

    # Define the order details
    order_details = {
        "orders": [
            {
                "conid": conid,
                "orderType": "MKT",
                "side": order,
                "tif": "DAY",
                "quantity": 1,
            }
        ]
    }

    response = requests.post(
        url=base_url + endpoint,
        json=order_details,
        verify=False
    )

    # Check for successful submission and extract the response data
    if response.status_code == 200:
        order_response = response.json()
        print("Order successfully placed:", order_response)
        return order_response
    else:
        print(f"Error placing order: {response.status_code} - {response.text}")
        response.raise_for_status()

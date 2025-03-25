from app.services.ibkr.connection import check_connection


class IBKRService:
    def __init__(self):
        # Initialize resources or setup here.
        pass

    def process_data(self, trading_data):
        print("trading_data", trading_data)

        # Check connection and halt execution immediately if it fails:
        check_connection()

        # Execution continues only if authentication succeeded.
        print("authenticated")

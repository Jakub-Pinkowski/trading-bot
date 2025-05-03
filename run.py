from flask import Flask

from config import DEBUG, PORT
from app.analysis.analysis_runner import run_analysis
from app.routes.webhook import webhook_blueprint
from app.services.ibkr.connection import start_ibkr_scheduler

app = Flask(__name__)
app.register_blueprint(webhook_blueprint)

# Tickle the API every 60 seconds
start_ibkr_scheduler()

# Run analysis if DEBUG is true
print(f"DEBUG: {DEBUG}")
# if DEBUG:
#     run_analysis()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)

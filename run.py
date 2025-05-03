from flask import Flask

from app.analysis.analysis_runner import run_analysis
from app.routes.webhook import webhook_blueprint
from app.services.ibkr.connection import start_ibkr_scheduler
from config import DEBUG, PORT

app = Flask(__name__)
app.register_blueprint(webhook_blueprint)

# Tickle the API every 60 seconds
start_ibkr_scheduler()

if __name__ == '__main__':
    # Run analysis only if in debug mode
    if DEBUG:
        run_analysis()

    app.run(host='0.0.0.0', port=PORT)

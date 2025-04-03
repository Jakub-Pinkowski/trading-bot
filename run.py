from flask import Flask

from app.analysis.analysis import get_recent_trades
from app.routes.webhook import webhook_blueprint
from app.services.ibkr.connection import start_ibkr_scheduler

app = Flask(__name__)
app.register_blueprint(webhook_blueprint)

# Tickle the API every 60 seconds
start_ibkr_scheduler()

# Testing
recent_trades = get_recent_trades()
print(recent_trades)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

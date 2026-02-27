from flask import Flask

from app.routes.webhook import webhook_blueprint
from config import PORT

app = Flask(__name__)
app.register_blueprint(webhook_blueprint)

# Tickle the API every 60 seconds
# start_ibkr_scheduler()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)

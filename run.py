from flask import Flask
from app.routes.webhook import webhook_blueprint

app = Flask(__name__)

app.register_blueprint(webhook_blueprint)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

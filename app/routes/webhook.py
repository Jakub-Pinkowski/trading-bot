from flask import Blueprint, request, abort

ALLOWED_IPS = {
    '52.89.214.238',
    '34.212.75.30',
    '54.218.53.128',
    '52.32.178.7',
    '127.0.0.1',
    'localhost'
}


def validate_ip(remote_addr):
    """Check if the sender IP is allowed."""
    if remote_addr not in ALLOWED_IPS:
        abort(403)


def parse_request_data(request):
    """Parse and return data from webhook request."""
    if request.content_type.startswith('application/json'):
        data = request.get_json()
        print("Received JSON data:", data)
    elif request.content_type.startswith('text/plain'):
        data = request.data.decode('utf-8')
        print("Received plain text:", data)
    else:
        abort(400, description='Unsupported Content-Type')

    return data


webhook_blueprint = Blueprint('webhook', __name__)


@webhook_blueprint.route('/webhook', methods=['POST'])
def webhook_route():
    """Full webhook handling logic."""
    validate_ip(request.remote_addr)
    data = parse_request_data(request)

    # TODO: Further processing of data here

    return '', 200
from flask import abort

ALLOWED_IPS = {
    '52.89.214.238',
    '34.212.75.30',
    '54.218.53.128',
    '52.32.178.7',
    '127.0.0.1',
    'localhost'
}


def validate_ip(remote_addr):
    if remote_addr not in ALLOWED_IPS:
        abort(403)


def parse_request_data(request):
    if request.content_type.startswith('application/json'):
        data = request.get_json()
    elif request.content_type.startswith('text/plain'):
        data = request.data.decode('utf-8')
    else:
        abort(400, description='Unsupported Content-Type')

    return data

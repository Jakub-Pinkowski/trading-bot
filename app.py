from flask import Flask, request, abort

app = Flask(__name__)

# Allowed IPs
ALLOWED_IPS = {
    '52.89.214.238',
    '34.212.75.30',
    '54.218.53.128',
    '52.32.178.7'
}


@app.before_request
def limit_remote_addr():
    allowed_testing_ips = {'127.0.0.1', 'localhost'}
    if request.remote_addr not in ALLOWED_IPS.union(allowed_testing_ips):
        abort(403)  # Forbidden access if IP is not allowed



@app.route('/webhook', methods=['POST'])
def webhook():
    if request.content_type.startswith('application/json'):
        data = request.get_json()
        print("Received JSON data:", data)
    elif request.content_type.startswith('text/plain'):
        data = request.data.decode('utf-8')
        print("Received plain text:", data)
    else:
        abort(400, description='Unsupported Content-Type')

    #TODO: Do the data processing here

    return '', 200  # Responding back with HTTP 200 OK status


# For testing purpose, use development server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)  # or use port 443 with SSL in production
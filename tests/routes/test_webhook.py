import unittest
from unittest.mock import patch

from flask import Flask
from werkzeug.exceptions import Forbidden

from app.routes.webhook import webhook_blueprint


class TestWebhookRoute(unittest.TestCase):

    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(webhook_blueprint)
        self.client = self.app.test_client()

    @patch('app.routes.webhook.validate_ip')
    @patch('app.routes.webhook.process_trading_data')
    def test_webhook_successful(self, mock_process_trading_data, mock_validate_ip):
        response = self.client.post('/webhook',
                                    json={"data": "valid"},
                                    headers={'Content-Type': 'application/json'},
                                    environ_base={'REMOTE_ADDR': '127.0.0.1'})

        mock_validate_ip.assert_called_once_with('127.0.0.1')
        mock_process_trading_data.assert_called_once_with({"data": "valid"})
        self.assertEqual(response.status_code, 200)

    @patch('app.routes.webhook.validate_ip')
    def test_webhook_unallowed_ip(self, mock_validate_ip):
        mock_validate_ip.side_effect = Forbidden(description='Forbidden IP')

        response = self.client.post('/webhook',
                                    json={"data": "valid"},
                                    headers={'Content-Type': 'application/json'},
                                    environ_base={'REMOTE_ADDR': '10.10.10.10'})

        mock_validate_ip.assert_called_once_with('10.10.10.10')
        self.assertEqual(response.status_code, 403)
        self.assertIn(b'Forbidden IP', response.data)

    @patch('app.routes.webhook.validate_ip')
    def test_webhook_bad_request_no_json(self, mock_validate_ip):
        response = self.client.post('/webhook',
                                    data="some data",
                                    headers={'Content-Type': 'text/plain'},
                                    environ_base={'REMOTE_ADDR': '127.0.0.1'})

        mock_validate_ip.assert_called_once_with('127.0.0.1')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Unsupported Content-Type', response.data)

    @patch('app.routes.webhook.validate_ip')
    @patch('app.routes.webhook.process_trading_data')
    def test_webhook_internal_server_error(self, mock_process_trading_data, mock_validate_ip):
        mock_process_trading_data.side_effect = Exception('Internal processing error')

        response = self.client.post('/webhook',
                                    json={"data": "valid"},
                                    headers={'Content-Type': 'application/json'},
                                    environ_base={'REMOTE_ADDR': '127.0.0.1'})

        mock_validate_ip.assert_called_once_with('127.0.0.1')
        mock_process_trading_data.assert_called_once_with({"data": "valid"})
        self.assertEqual(response.status_code, 500)
        self.assertIn(b'Internal processing error', response.data)


if __name__ == '__main__':
    unittest.main()

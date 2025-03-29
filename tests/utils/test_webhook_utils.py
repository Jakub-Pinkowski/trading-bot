import unittest
from unittest.mock import Mock, patch

from flask import Flask
from werkzeug.exceptions import HTTPException

from app.utils.webhook_utils import validate_ip, parse_request_data


class TestWebhookUtils(unittest.TestCase):

    @patch('app.utils.webhook_utils.abort')
    def test_validate_ip_allowed(self, mock_abort):
        allowed_ips = [
            '52.89.214.238',
            'localhost',
            '127.0.0.1',
            '54.218.53.128'
        ]

        for ip in allowed_ips:
            validate_ip(ip)
            mock_abort.assert_not_called()

    @patch('app.utils.webhook_utils.abort')
    def test_validate_ip_disallowed(self, mock_abort):
        disallowed_ip = '10.10.10.10'
        validate_ip(disallowed_ip)
        mock_abort.assert_called_once_with(403)

    def setUp(self):
        self.app = Flask(__name__)

    def test_parse_request_data_json(self):
        mock_request = Mock(content_type='application/json', get_json=Mock(return_value={"key": "value"}))
        with self.app.test_request_context(json={'key': 'value'}):
            data = parse_request_data(mock_request)
            self.assertEqual(data, {'key': 'value'})

    @patch('app.utils.webhook_utils.abort')
    def test_parse_request_data_unsupported_content_type(self, mock_abort):
        mock_abort.side_effect = HTTPException(description='Unsupported Content-Type', response=Mock(status=400))
        mock_request = Mock(content_type='application/x-www-form-urlencoded')

        with self.assertRaises(HTTPException) as context:
            parse_request_data(mock_request)

        mock_abort.assert_called_once_with(400, description='Unsupported Content-Type')
        self.assertEqual(context.exception.response.status, 400)


if __name__ == '__main__':
    unittest.main()

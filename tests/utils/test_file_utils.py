import json
from unittest.mock import mock_open, patch, ANY

from app.utils.file_utils import load_file, save_file


@patch('app.utils.file_utils.os.path.exists', return_value=True)
@patch('builtins.open', new_callable=mock_open, read_data='{"test_key": "test_value"}')
def test_load_cache_file_exists(mock_file, mock_exists):
    expected_data = {"test_key": "test_value"}
    result = load_file()

    assert result == expected_data
    mock_file.assert_called_once_with(ANY, 'r')


@patch('app.utils.file_utils.os.path.exists', return_value=False)
def test_load_cache_file_missing(mock_exists):
    result = load_file()

    assert result == {}


@patch('builtins.open', new_callable=mock_open)
def test_save_cache(mock_file):
    data_to_save = {"key": "value"}
    save_file(data_to_save)

    mock_file.assert_called_once_with(ANY, 'w')
    handle = mock_file()
    handle.write.assert_called()

    written_content = ''.join(call.args[0] for call in handle.write.call_args_list)
    assert json.loads(written_content) == data_to_save

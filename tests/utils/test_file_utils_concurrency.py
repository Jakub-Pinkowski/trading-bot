import multiprocessing
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.utils.file_utils import save_to_parquet
from config import BACKTESTING_DIR


def worker_process(worker_id, file_path, iterations=5):
    """Worker process that writes to a parquet file concurrently."""
    for i in range(iterations):
        # Create a DataFrame with unique data for this worker
        data = pd.DataFrame({
            'worker_id': [worker_id] * 3,
            'iteration': [i] * 3,
            'value': [worker_id * 100 + i * 10 + j for j in range(3)]
        })
        
        # Save to parquet - this should be thread-safe with file locking
        save_to_parquet(data, file_path)
        
        # Small delay to increase chance of collision
        time.sleep(0.01)
    
    return worker_id


def check_parquet_integrity(file_path):
    """Check if the parquet file is valid by trying to load it."""
    try:
        df = pd.read_parquet(file_path)
        if isinstance(df, pd.DataFrame):
            return True, len(df)
        else:
            return False, 0
    except Exception as e:
        print(f"Error loading parquet: {e}")
        return False, 0


@pytest.fixture
def clean_test_parquet():
    """Fixture to ensure a clean test environment."""
    # Ensure the backtesting directory exists
    Path(BACKTESTING_DIR).mkdir(parents=True, exist_ok=True)
    
    # Generate a unique file name for this test run
    file_name = f"concurrency_test_{int(time.time() * 1000)}.parquet"
    file_path = os.path.join(BACKTESTING_DIR, file_name)
    
    # Remove any existing test file
    if os.path.exists(file_path):
        os.remove(file_path)
    
    yield file_path
    
    # Cleanup after test
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Clean up lock file
    lock_file = f"{file_path}.lock"
    if os.path.exists(lock_file):
        os.remove(lock_file)


def test_save_to_parquet_concurrent_access(clean_test_parquet):
    """Test that save_to_parquet handles concurrent access correctly."""
    file_path = clean_test_parquet
    num_workers = 4
    iterations = 5
    
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Start the worker processes
        results = pool.starmap(worker_process, [(i, file_path, iterations) for i in range(num_workers)])
    
    # Verify all workers completed successfully
    assert len(results) == num_workers
    for i in range(num_workers):
        assert i in results
    
    # Check parquet file integrity
    is_valid, num_rows = check_parquet_integrity(file_path)
    assert is_valid, "Parquet file is corrupted or invalid"
    assert num_rows > 0, "Parquet file is empty"
    
    # Load the parquet file and verify its contents
    df = pd.read_parquet(file_path)
    
    # We should have deduplicated data from all workers
    # Each worker writes 3 rows per iteration
    # With deduplication, we might have fewer rows
    assert len(df) > 0, "DataFrame is empty"
    
    # Verify the DataFrame has the expected columns
    assert 'worker_id' in df.columns
    assert 'iteration' in df.columns
    assert 'value' in df.columns
    
    # Verify worker_ids are in expected range
    unique_workers = df['worker_id'].unique()
    assert len(unique_workers) <= num_workers
    for worker_id in unique_workers:
        assert 0 <= worker_id < num_workers
    
    # Verify data integrity - all rows should be valid
    assert df['worker_id'].notna().all(), "Found NaN in worker_id column"
    assert df['iteration'].notna().all(), "Found NaN in iteration column"
    assert df['value'].notna().all(), "Found NaN in value column"


def test_save_to_parquet_sequential_consistency(clean_test_parquet):
    """Test that sequential writes produce consistent results."""
    file_path = clean_test_parquet
    
    # First write
    df1 = pd.DataFrame({
        'id': [1, 2, 3],
        'value': [10, 20, 30]
    })
    save_to_parquet(df1, file_path)
    
    # Verify first write
    result1 = pd.read_parquet(file_path)
    assert len(result1) == 3
    
    # Second write with different data
    df2 = pd.DataFrame({
        'id': [4, 5],
        'value': [40, 50]
    })
    save_to_parquet(df2, file_path)
    
    # Verify concatenation and deduplication
    result2 = pd.read_parquet(file_path)
    assert len(result2) == 5  # Should have all 5 unique rows
    
    # Third write with duplicate data
    df3 = pd.DataFrame({
        'id': [1, 2],  # Duplicates from first write
        'value': [10, 20]
    })
    save_to_parquet(df3, file_path)
    
    # Verify deduplication worked
    result3 = pd.read_parquet(file_path)
    assert len(result3) == 5  # Should still have 5 unique rows


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

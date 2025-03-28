#!/usr/bin/env python3

import datetime
from huggingface_hub import HfApi
import bittensor as bt
import time
# Bittensor constants
BLOCK_DURATION = 12  # seconds

def get_model_info(repo_id):
    """
    Get detailed information about a Hugging Face model
    
    Args:
        repo_id (str): Repository ID in format "username/model_name"
        
    Returns:
        tuple: (model_info, creation_date) or (None, None) if error
    """
    try:
        # Create HfApi client
        api = HfApi()
        
        # Get model info
        model_info = api.model_info(repo_id)
        # Extract creation date
        creation_date = model_info.created_at if hasattr(model_info, 'created_at') else None
        
        return model_info, creation_date
        
    except Exception as e:
        bt.logging.error(f"Error getting model info for {repo_id}: {e}")
        return None, None

def block_to_time(block):
    """
    Convert a Bittensor block number to actual time in UTC.
    
    Args:
        block (int): The block number to convert
    
    Returns:
        datetime: The UTC datetime corresponding to the given block
    """
    current_block = bt.subtensor().get_current_block()
    current_time = datetime.datetime.now(datetime.timezone.utc)
    
    # Calculate seconds since the block
    seconds_diff = (current_block - block) * BLOCK_DURATION
    
    # Calculate the datetime for the requested block
    block_time = current_time - datetime.timedelta(seconds=seconds_diff)
    return block_time

def compare_block_and_model(block_number, repo_id):
    """
    Compare a Bittensor block time with a Hugging Face model creation date.
    
    Args:
        block_number (int): Bittensor block number
        repo_id (str): Hugging Face repository ID in format "username/model_name"
        
    Returns:
        bool: True if block time is later than model creation time, False otherwise
    """
    # Get block time
    block_time = block_to_time(block_number)
    
    # Get model creation time
    _, model_creation_time = get_model_info(repo_id)
    time.sleep(1)
    if model_creation_time is None:
        return False
    
    # Calculate time difference and return whether block is later than model creation
    time_difference = block_time - model_creation_time if isinstance(model_creation_time, datetime.datetime) else None
    return time_difference.total_seconds() > BLOCK_DURATION
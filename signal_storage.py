#!/usr/bin/env python3
"""
Signal Storage Module for DataNexus
Shared storage for provisional trading signals between modules.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional

# Global storage for provisional signals
provisional_signals: Dict[str, Dict[str, Any]] = {}


def store_signal(signal_data: Dict[str, Any]) -> str:
    """
    Store a trading signal for verification.
    
    Args:
        signal_data: Dictionary containing signal information
        
    Returns:
        Generated signal ID
    """
    signal_id = str(uuid.uuid4())
    signal_data['signal_id'] = signal_id
    signal_data['created_at'] = datetime.now().isoformat()
    signal_data['status'] = 'PENDING'
    
    provisional_signals[signal_id] = signal_data
    return signal_id


def get_signal(signal_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a signal by ID.
    
    Args:
        signal_id: Signal identifier
        
    Returns:
        Signal data or None if not found
    """
    return provisional_signals.get(signal_id)


def get_all_signals() -> Dict[str, Any]:
    """
    Get all provisional signals.
    
    Returns:
        Dictionary with signals list and count
    """
    return {
        'signals': list(provisional_signals.values()),
        'count': len(provisional_signals)
    }


def update_signal_status(signal_id: str, status: str, **kwargs) -> bool:
    """
    Update signal status and additional fields.
    
    Args:
        signal_id: Signal identifier
        status: New status
        **kwargs: Additional fields to update
        
    Returns:
        True if updated successfully, False if signal not found
    """
    if signal_id not in provisional_signals:
        return False
        
    provisional_signals[signal_id]['status'] = status
    provisional_signals[signal_id]['updated_at'] = datetime.now().isoformat()
    
    # Update additional fields
    for key, value in kwargs.items():
        provisional_signals[signal_id][key] = value
        
    return True


def clear_old_signals(max_age_hours: int = 24):
    """
    Clear signals older than specified hours.
    
    Args:
        max_age_hours: Maximum age in hours
    """
    cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
    
    to_remove = []
    for signal_id, signal in provisional_signals.items():
        signal_time = datetime.fromisoformat(signal['created_at']).timestamp()
        if signal_time < cutoff_time:
            to_remove.append(signal_id)
    
    for signal_id in to_remove:
        del provisional_signals[signal_id]
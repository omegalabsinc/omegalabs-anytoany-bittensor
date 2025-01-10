import os
import pwd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def verify_sandbox_context():
    """Verify we're running in sandbox context"""
    try:
        sandbox_uid = pwd.getpwnam("sandbox").pw_uid
        current_uid = os.getuid()
        
        if current_uid != sandbox_uid:
            raise RuntimeError("Must run as sandbox user")
            
        if os.access("/etc/passwd", os.W_OK):
            raise RuntimeError("Has write access outside sandbox")
            
        if not os.path.exists("/sandbox"):
            raise RuntimeError("Sandbox directory missing")
            
        return True
    except Exception as e:
        logger.error(f"Sandbox verification failed: {e}")
        return False
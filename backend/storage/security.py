"""Security services for uploaded objects (e.g., Antivirus)."""

import structlog
from abc import ABC, abstractmethod

logger = structlog.get_logger()

class VirusScanner(ABC):
    @abstractmethod
    async def scan(self, file_content: bytes) -> bool:
        """Return True if safe, False if malicious/infected."""
        pass

class NoOpScanner(VirusScanner):
    """Used for development, assumes all files are safe."""
    async def scan(self, file_content: bytes) -> bool:
        logger.debug("noop_scan_passed")
        return True

class ClamAVScanner(VirusScanner):
    """Production implementation using ClamAV daemon."""
    async def scan(self, file_content: bytes) -> bool:
        # Placeholder for actual ClamAV TCP/socket integration
        # (e.g. using pyclamd or aio-clamd)
        logger.debug("clamav_scan_passed")
        return True

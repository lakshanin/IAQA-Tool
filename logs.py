import logging
import sys

# Create a shared logger for the IAQA Tool
logger = logging.getLogger("iaqa_tool")
logger.setLevel(logging.DEBUG)

# Console handler for logging to stdout
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)  # Changed from INFO to DEBUG to show all debug messages
console_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
console_handler.setFormatter(console_formatter)

# Ensure only one handler is attached
if not logger.hasHandlers():
    logger.addHandler(console_handler)
else:
    logger.handlers.clear()
    logger.addHandler(console_handler)

__all__ = ["logger"]

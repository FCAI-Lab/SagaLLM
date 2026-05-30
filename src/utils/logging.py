"""
logging.py — Console Print Utilities
=====================================
Lightweight helpers for formatted console output used during pipeline execution.
"""

import time

from colorama import Fore, Style


def custom_print(message: str) -> None:
    """Print a highlighted banner message with a short delay for readability."""
    print(Style.BRIGHT + Fore.CYAN + f"\n{'=' * 50}")
    print(Fore.MAGENTA + f"{message}")
    print(Style.BRIGHT + Fore.CYAN + f"{'=' * 50}\n")
    time.sleep(0.5)


def custom_step_tracker(step: int, total_steps: int) -> None:
    """Print a 'STEP N/M' progress banner."""
    custom_print(f"STEP {step + 1}/{total_steps}")

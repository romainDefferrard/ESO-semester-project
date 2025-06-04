"""
Filename: timer_logger.py
Author: Romain Defferrard
Date: 04-06-2025
"""
import time
import logging

class TimerLogger:
    """
    Helper class for timing and logging the duration of code blocks.

    This tool allows the user to mark the start and end of labeled code segments, 
    accumulate their execution durations, and print out a summary of all timed blocks.
    """
    def __init__(self):
        self.times = {}
        self._start_times = {}

    def start(self, label: str) -> None:
        """
        Start timing a labeled code block.

        Input:
            label (str): Identifier for the timed block.
        
        Output:
            None
        """
        self._start_times[label] = time.time()

    def stop(self, label: str) -> None:
        """
        Stop timing a labeled code block and log the elapsed time.

        Input:
            label (str): Identifier for the timed block.
        
        Output:
            None

        Logs a warning if no start time was recorded for the given label.
        """
        if label not in self._start_times:
            logging.warning(f"No start time found for label '{label}'")
            return
        elapsed = time.time() - self._start_times.pop(label)
        self.times[label] = self.times.get(label, 0.0) + elapsed
        logging.info(f"{label}: {elapsed:.2f}s")

    def summary(self) -> None:
        """
        Log a summary of total accumulated times for all recorded labels.

        Output:
            None
        """
        logging.info("\n=== Timing Summary ===")
        for label, total in self.times.items():
            logging.info(f"{label:<30} {total:.2f}s")

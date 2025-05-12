import time
import logging

class TimerLogger:
    def __init__(self):
        self.times = {}
        self._start_times = {}

    def start(self, label: str):
        self._start_times[label] = time.time()

    def stop(self, label: str):
        if label not in self._start_times:
            logging.warning(f"No start time found for label '{label}'")
            return
        elapsed = time.time() - self._start_times.pop(label)
        self.times[label] = self.times.get(label, 0.0) + elapsed
        logging.info(f"{label}: {elapsed:.2f}s")

    def summary(self):
        logging.info("\n=== Timing Summary ===")
        for label, total in self.times.items():
            logging.info(f"{label:<30} {total:.2f}s")

import json
import pandas as pd
import os
import numpy as np


class FlightData:
    def __init__(self, config):

        self.las_dir = config["LAS_DIR"]
        self.log_dir = config["LOG_DIR"]
        self.trajectory_path = config["TRAJECTORY_PATH"]
        self.dataset_name = config["DATASET_NAME"]

        self.flights = {}  # Store extracted flights
        self.bounds = []  # Store E/N - min/max coordinates for each flight
        self.center = []  # center of mass E,N

        # Extract flight_times
        self.flight_times = self.extract_flight_times()

        # Load full flight data from csv/txt
        self.flight_df = self.load_flight_data()

        # Load time intervals and extract flights
        self.load_flights()

    def extract_flight_times(self):
        if self.dataset_name == "Arpette":
            self.flight_times = self.extract_flight_times_arpette()
        elif self.dataset_name == "Vallet":
            self.flight_times = self.extract_flight_times_vallet()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        return self.flight_times

    def extract_flight_times_arpette(self):
        """
        Extract flight start and end times from corresponding .sdc.log files.

        :return: Dictionary with flight IDs as keys and (start, end) timestamps as values.
        """
        flight_times = {}
        flight_ids = []

        directory = os.path.dirname(self.las_dir)
        for filename in os.listdir(directory):
            if filename.endswith(".laz"):
                flight_name = filename.split(".")[0]  # Extract numeric part before file format
                flight_id = flight_name.split("_")[-1]
                flight_ids.append(flight_id)

        flight_ids.sort()

        log_file_pattern = os.path.basename(self.log_dir)
        directory_log = os.path.join(directory, "timestamps")
        for flight_id in flight_ids:
            log_file = log_file_pattern.format(flight_id=flight_id)

            log_file_path = os.path.join(directory_log, log_file)

            if os.path.exists(log_file_path):
                with open(log_file_path, "r", encoding="ISO-8859-1") as f:
                    lines = f.readlines()

                start_time = None
                end_time = None

                # Extract start time from line 137
                start_time_line = lines[136].strip()
                if "File start" in start_time_line:
                    start_time = float(start_time_line.split("(")[1].split()[0])  # A voir si on garde juste un int() ou float()

                # Extract end time from line 138
                end_time_line = lines[137].strip()
                if "File end" in end_time_line:
                    end_time = float(end_time_line.split("(")[1].split()[0])  # Extract time before 's'

                if start_time is not None and end_time is not None:
                    flight_times[flight_id] = {"start": start_time, "end": end_time}

        return flight_times

    def extract_flight_times_vallet(self):
        """Load flight timestamps from `GPS_Times.json` for Vallet dataset."""
        if not os.path.exists(self.log_dir):
            raise FileNotFoundError(f"GPS_Times.json not found at {self.log_dir}")

        with open(self.log_dir, "r") as f:
            flight_times_json = json.load(f)

        flight_times = {}
        for flight_name, times in flight_times_json["flight_intervals"].items():
            try:
                flight_id = flight_name.split("_")[-1]
                start_time = float(times["start_time"])
                end_time = float(times["end_time"])
                flight_times[flight_id] = {"start": start_time, "end": end_time}
            except (KeyError, ValueError):
                print(f"Warning: Missing or invalid time data for flight {flight_id}")

        return flight_times

    def load_flight_data(self):
        if self.dataset_name == "Arpette":
            cols = ["gps_time", "lon", "lat", "alt", "roll", "pitch", "yaw", "?"]
        elif self.dataset_name == "Vallet":
            cols = [
                "gps_time",
                "lon",
                "lat",
                "alt",
                "roll",
                "pitch",
                "yaw",
            ]

        return pd.read_csv(self.trajectory_path, names=cols, header=None)

    def load_flights(self):

        all_flight_data = []
        if self.dataset_name == "Arpette":
            day_of_week = 3
        elif self.dataset_name == "Vallet":
            day_of_week = 0

        for flight_id, interval in self.flight_times.items():
            start, end = int(interval["start"]) + day_of_week * 24 * 3600, int(interval["end"]) + day_of_week * 24 * 3600  # Change depending on the DOW

            flight_data = self.flight_df[(self.flight_df["gps_time"] >= start) & (self.flight_df["gps_time"] <= end)]
            flight_name = f"Flight_{flight_id}"  # Create a name based on flight_id

            self.flights[flight_name] = flight_data

            all_flight_data.append(flight_data)

        # Storing bounds E_min, E_max, N_min, N_max format
        self.compute_flight_bounds(all_flight_data)

    def compute_flight_bounds(self, flight_data):
        combined_data = pd.concat(flight_data, ignore_index=True)
        E_min, E_max = combined_data["lon"].min(), combined_data["lon"].max()
        N_min, N_max = combined_data["lat"].min(), combined_data["lat"].max()
        self.bounds = [E_min, E_max, N_min, N_max]

"""
Filename: flight_data.py
Author: Romain Defferrard
Date: 08-05-2025

Description:
    This script handles the extraction and organization of flight data from trajectory files.
    It sequences the flight lines from time intervals and the bounds coordinates of all flights 
    are computed for further processing of DTM loading

    Main features:
    - Extract flight start/end times from .log or JSON.
    - Load trajectory data 
    - Slice trajectories into individual flights.
    - Compute spatial bounds of all flight data.
"""
import json
import pandas as pd
import os
import numpy as np
import logging


class FlightData:
    def __init__(self, config):
        """
        Initializes the FlightData object with configuration and loads flight data.

        Input:
            config (dict): Configuration dictionary.
                - LAS_DIR (str): Template path to LAS/LAZ or ASCII files.
                - LOG_DIR (str): Path to log files or JSON metadata.
                - TRAJECTORY_PATH (str): Path to the full trajectory .csv/.txt file.
                - DAY_OF_WEEK (int): Day offset used for GPS time correction.

        Output:
            - self.flights (dict[str, pd.DataFrame]): Flight data per ID.
            - self.bounds (list[float]): Combined E/N bounds across all flights.
        """
        self.las_dir = config["LAS_DIR"]
        self.log_dir = config["LOG_DIR"]
        self.trajectory_path = config["TRAJECTORY_PATH"]
        self.dow = config["DAY_OF_WEEK"]
        self.trajectory_cols = config["TRAJECTORY_COLUMNS"]

        self.flights = {}  # Store extracted flights
        self.bounds = []  # Store E/N - min/max coordinates for each flight

        # Extract flight_times
        self.flight_times = self.extract_flight_times()

        # Load full flight data 
        self.flight_df = self.load_flight_data()

        # Load time intervals and extract flights
        self.load_flights()

    
    def extract_flight_times(self):
        """
        Extracts flight start/end times for each flight ID from timing metadata.

        If a JSON file exists at self.log_dir (recommended), it is used to extract 
        start and end times for each flight. Otherwise, a custom .log file method 
        is used (used in Arpette dataset).

        Returns:
            dict[str, dict[str, float]]: Flight time intervals per flight ID, formatted as:
                {"flight_id": {"start": float, "end": float}, ...}

        Raises:
            FileNotFoundError: If neither JSON nor log files are found.
            ValueError: If file format or parsing fails.
        """
        # First case: recommended -> JSON format
        if os.path.exists(self.log_dir) and self.log_dir.endswith(".json"):
            with open(self.log_dir, "r") as f:
                flight_times_json = json.load(f)

            flight_times = {}
            for flight_name, times in flight_times_json.get("flight_intervals", {}).items():
                try:
                    flight_id = flight_name.split("_")[-1]
                    start_time = float(times["start_time"])
                    end_time = float(times["end_time"])
                    flight_times[flight_id] = {"start": start_time, "end": end_time}
                except (KeyError, ValueError) as e:
                    logging.warning(f"Invalid or missing time data for flight '{flight_name}': {e}")

            return flight_times

        # Second case: legacy Arpette dataset -> .log files per flight
        flight_times = {}
        flight_ids = []

        directory = os.path.dirname(self.las_dir)
        for filename in os.listdir(directory):
            if filename.endswith(".laz") or filename.endswith(".las") or filename.endswith(".txt") or filename.endswith(".TXYZS"):
                flight_name = filename.split(".")[0]
                flight_id = flight_name.split("_")[-1]
                flight_ids.append(flight_id)

        flight_ids.sort()
        directory_log = os.path.join(directory, "timestamps")
        log_file_pattern = os.path.basename(self.log_dir)

        for flight_id in flight_ids:
            log_file = log_file_pattern.format(flight_id=flight_id)
            log_file_path = os.path.join(directory_log, log_file)

            if not os.path.exists(log_file_path):
                continue
            with open(log_file_path, "r", encoding="ISO-8859-1") as f:
                lines = f.readlines()

            try:
                start_time_line = lines[136].strip()
                end_time_line = lines[137].strip()
                if "File start" in start_time_line and "File end" in end_time_line:
                    start_time = float(start_time_line.split("(")[1].split()[0])
                    end_time = float(end_time_line.split("(")[1].split()[0])
                    flight_times[flight_id] = {"start": start_time, "end": end_time}
                else:
                    logging.warning(f"Unexpected format in log file: {log_file_path}")
            except (ValueError, IndexError) as e:
                logging.warning(f"Failed to parse times from {log_file_path}: {e}")

        if not flight_times:
            raise FileNotFoundError("No valid flight times found in either JSON or log files.")

        return flight_times

    
    def load_flight_data(self):
        """
        Loads the full trajectory data file as a pandas DataFrame.

        Uses the column names defined in the configuration file.

        Returns:
            pd.DataFrame: Full trajectory data.
        """
        cols = self.trajectory_cols
        return pd.read_csv(self.trajectory_path, names=cols, header=None)

    def load_flights(self):
        """
        Filters the full trajectory data per extracted flight time intervals.

        Input:
            None

        Output:
            None, but populates self.flights dictionary and updates self.bounds.
        """
        all_flight_data = []

        for flight_id, interval in self.flight_times.items():
            # Change depending on the dow (Day Of Week)
            start, end = int(interval["start"]) + self.dow * 24 * 3600, int(interval["end"]) + self.dow * 24 * 3600  
            flight_data = self.flight_df[(self.flight_df["gps_time"] >= start) & (self.flight_df["gps_time"] <= end)]
            flight_name = f"Flight_{flight_id}"  # Create a name based on flight_id

            self.flights[flight_name] = flight_data

            all_flight_data.append(flight_data)

        # Storing bounds E_min, E_max, N_min, N_max format
        self.compute_flight_bounds(all_flight_data)

    def compute_flight_bounds(self, flight_data):
        """
        Computes overall E/N bounds from concatenated flight data.

        Input:
            flight_data (list[pd.DataFrame]): List of flight-specific DataFrames.

        Output:
            None (updates self.bounds)
        """
        combined_data = pd.concat(flight_data, ignore_index=True)
        E_min, E_max = combined_data["lon"].min(), combined_data["lon"].max()
        N_min, N_max = combined_data["lat"].min(), combined_data["lat"].max()
        self.bounds = [E_min, E_max, N_min, N_max]

import json
import pandas as pd
import os
import numpy as np
import logging


class FlightData:
    def __init__(self, config):

        self.las_dir = config["LAS_DIR"]
        self.log_dir = config["LOG_DIR"]
        self.trajectory_path = config["TRAJECTORY_PATH"]
        #self.dataset_name = config["DATASET_NAME"]
        self.dow = config["DAY_OF_WEEK"]

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
        """
        Extract flight start and end times:
        - From .sdc.log files for .las/.laz
        - From GPS_Times.json for .txt/.TXYZS

        :return: Dictionary {flight_id: {"start": ..., "end": ...}}
        :raises ValueError: if file format is unknown
        """
        # Determine file format
        if "." not in self.las_dir:
            raise ValueError(f"Unable to determine file format from path: {self.las_dir}")
        
        self.file_format = self.las_dir.split(".")[-1].lower()

        if self.file_format in ["las", "laz"]:
            flight_times = {}
            flight_ids = []

            directory = os.path.dirname(self.las_dir)
            for filename in os.listdir(directory):
                if filename.endswith(".laz"):
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
                    logging.warning(f"Missing log file: {log_file_path}")
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
                except IndexError:
                    logging.warning(f"Log file too short or malformed: {log_file_path}")
                except (ValueError, IndexError) as e:
                    logging.warning(f"Failed to parse times from {log_file_path}: {e}")

            return flight_times

        elif self.file_format in ["txt", "TXYZS"]:
            if not os.path.exists(self.log_dir):
                raise FileNotFoundError(f"GPS_Times.json not found at {self.log_dir}")

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

        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
            
    
    def load_flight_data(self):
        if self.file_format in ["las", "laz"]:
            cols = ["gps_time", "lon", "lat", "alt", "roll", "pitch", "yaw", "?"]
        else:
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

        for flight_id, interval in self.flight_times.items():
            start, end = int(interval["start"]) + self.dow * 24 * 3600, int(interval["end"]) + self.dow * 24 * 3600  # Change depending on the DOW
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

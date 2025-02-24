import json
import pandas as pd

class FlightData:
    def __init__(self, config_path, csv_path):
        self.config_path = config_path

        self.csv_path = csv_path
        self.flights = {}  # Store extracted flights
        self.bounds = [] # Store E/N - min/max coordinates for each flight
        self.center = [] # center of mass E,N
        # Load full flight data from CSV
        self.flight_df = self.load_flight_data()
        
        # Load time intervals and extract flights
        self.load_flights()

    def load_flight_data(self):

        cols = ['gps_time', 'lon', 'lat', 'alt', 'roll', 'pitch', 'yaw']
        return pd.read_csv(self.csv_path, names=cols, header=None)

    def load_flights(self):

        with open(self.config_path, 'r') as file:
            times = json.load(file)

        all_flight_data = [] 
        
        for flight_name, interval in times["flight_intervals"].items():
            # check len(flight_name) == len(flight_df)
            start, end = interval["start"], interval["end"]
            flight_data = self.flight_df[
                (self.flight_df["gps_time"] >= start) & (self.flight_df["gps_time"] <= end)
            ]
            self.flights[flight_name] = flight_data

            all_flight_data.append(flight_data)
            
        # Storing bounds E_min, E_max, N_min, N_max format
        self.compute_flight_bounds(all_flight_data)
        # Get center 
        self.compute_center_of_mass(all_flight_data)
      
        
    def compute_flight_bounds(self, flight_data):
        combined_data = pd.concat(flight_data, ignore_index=True)
        E_min, E_max = combined_data['lon'].min(), combined_data['lon'].max()
        N_min, N_max = combined_data['lat'].min(), combined_data['lat'].max()
        self.bounds = [E_min, E_max, N_min, N_max]

    # Not useful anymore (?)
    def compute_center_of_mass(self, all_flight_data):
        combined_data = pd.concat(all_flight_data, ignore_index=True)
        E_center = int(combined_data['lon'].mean())
        N_center = int(combined_data['lat'].mean())
        self.center = (E_center, N_center)

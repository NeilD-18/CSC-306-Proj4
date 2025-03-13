import os
import csv
import sys
import argparse

class DataAgent:
    """
    A class to handle loading and managing CSV data from competition directories.
    The data is stored in a nested dictionary structure organized by competition folders.
    """
    
    def __init__(self):
        """Initialize an empty DataAgent with a dictionary to store the loaded data."""
        self.data = {}

    def extract_column_names(self, data):
        """
        Extract column names from the first row of CSV data.
        
        Args:
            data (list): List of CSV rows where the first row contains column names
            
        Returns:
            list: Column names from the first row, or empty list if data is empty
        """
        if data and len(data) > 0:
            return data[0]  # First row contains column names
        return []

    def load_data(self, competition_dir):
        """
        Load CSV data from a competition directory structure.
        
        Args:
            competition_dir (str): Path to the directory containing competition folders
            
        Each competition folder should contain:
            - sample.csv: Sample dataset
            - all.csv: Complete dataset
            
        The data is stored in self.data as:
            {competition_folder: {
                'sample': [...],
                'sample_column_names': [...],
                'all': [...],
                'all_column_names': [...]
            }}
        """
        if not os.path.exists(competition_dir):
            print(f"Error: Competition directory not found: {competition_dir}")
            return False
            
        try:
            for folder_name in os.listdir(competition_dir):
                folder_path = os.path.join(competition_dir, folder_name)
                if os.path.isdir(folder_path):
                    self.data[folder_name] = {}
                    sample_csv_path = os.path.join(folder_path, 'sample.csv')
                    all_csv_path = os.path.join(folder_path, 'all.csv')
                    
                    if os.path.exists(sample_csv_path):
                        with open(sample_csv_path, 'r') as sample_file:
                            sample_data = list(csv.reader(sample_file))
                            self.data[folder_name]['sample'] = sample_data
                            self.data[folder_name]['sample_column_names'] = self.extract_column_names(sample_data)
                    
                    if os.path.exists(all_csv_path):
                        with open(all_csv_path, 'r') as all_file:
                            all_data = list(csv.reader(all_file))
                            self.data[folder_name]['all'] = all_data
                            self.data[folder_name]['all_column_names'] = self.extract_column_names(all_data)
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def print_dictionary_keys(self, d, indent=0):
        """
        Recursively print all keys in a nested dictionary structure.
        
        Args:
            d (dict): Dictionary to print keys from
            indent (int): Current indentation level for pretty printing
        """
        for key, value in d.items():
            print("  " * indent + f"Key: {key}")
            if isinstance(value, dict):
                self.print_dictionary_keys(value, indent + 1)

# Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and analyze competition data.')
    parser.add_argument('--data-dir', type=str, 
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'competition'),
                        help='Path to the competition directory (default: ../competition)')
    
    args = parser.parse_args()
    
    agent = DataAgent()
    competition_directory = args.data_dir
    
    print(f"Attempting to load data from: {competition_directory}")
    success = agent.load_data(competition_directory)
    
    if success:
        print("\nAll dictionary keys:")
        agent.print_dictionary_keys(agent.data)
    else:
        print("\nFailed to load data. Please verify the directory path.")
        print("You can specify a different path with --data-dir argument.")
        print("Example: python dataAgent.py --data-dir /path/to/competition/directory")

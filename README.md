### Project 4 

### dataAgent.py
The dataAgent class is responsible for managing and processing datasets from various sources. 
It provides functionalities to handle data samples, retrieve column names, and access complete datasets.

## Populate Dictionary data
To load the datasets into a dicitonary, run the following

```
agent = DataAgent()
competition_directory = os.path.join(os.path.dirname(__file__), 'competition')
agent.load_data(competition_directory)
```

Dictionary Structure:
- 066_IBM_HR:
    - sample: Contains a sample of the IBM HR dataset
    - sample_column_names: List of column names for the sample of the IBM HR dataset.
    - all: Contains the entire IBM HR dataset.
    - all_column_names: List of column names for the entire IBM HR dataset.
- 067_TripAdvisor:
    - sample: Contains a sample of the TripAdvisor dataset.
    - sample_column_names: List of column names for the sample of the TripAdvisor dataset.
    - all: Contains the entire TripAdvisor dataset.
    - all_column_names: List of column names for the entire TripAdvisor dataset.
...

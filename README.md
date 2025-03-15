# CSC306 Final Project - Question Answering Ovet Tabular Data

This project involves various techniques to solve questions over tabular data using different LLM models.

## Environment Setup

1. Create a virtual environment:
   ```sh
   python -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:
     ```sh
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Folder Structure

- models: Contains different LLM models built using various techniques to solve questions over tabular data.
  - `zero_shot_incontext_learning.py`
  - `zero_shot_icl_2.py`
  - `zero_shot_baseline.py`
  - `prompt_engineering.py`
  - `cot_prompting.py`
  - `code_based_learning.py`

- agents: Contains agents used for data handling and evaluation.
  - `dataAgent.py`: Handles loading and managing CSV data from competition directories.
  - `eval_agent.py`: Evaluates the models using the provided datasets.
  - `default_comparer_agent.py`: Compares model responses to actual answers using a default comparison method.

- papers: Contains documents needed for the project such as the research paper it is based on along with the project's description.
  - `Question_Answering_Over_Tabular_Data_with_DataBeach.pdf`
  - `Project_Proposal.pdf`

- model_responses: Contains the responses of each model to the questions dataset.
  - `responses_zero_shot_icl.txt`
  - `responses_zero_shot_CoT.txt`
  - `responses_zero_shot_baseline_4o-mini.txt`
  - `responses_zero_shot_baseline_3.5-turbo.txt`
  - `responses_PE.txt`
  - `responses_pe_4o-mini.txt`
  - `responses_cbl_4o-mini.txt`
  - `responses_cbl.txt`
  - `responses_icl_4o-mini.txt`
  - `responses_Cot_4o-mini.txt`
  - `responses_Cot_3.5-turbo.txt`

- answers: Contains the answers to the questions, both for all and sample data.
  - `answers.txt`
  - `answers_lite.txt`
  - `semantics.txt`

## Usage

For each of the following modules you can specify responses or even answers by changing the inputs inside the main of the respective module.

To evaluate a model, you can use the `eval_agent.py` script. For example:
```sh
python agents/eval_agent.py
```

This will run the evaluation using the specified model and save the responses to a file. All models currently implemented have been initialized. To select which model to evaluate, provide it in the model argument and give the save path the name of the responses.txt file that it will output.

To compare the model responses to the actual answers using the default comparison method, you can use the `default_comparer_agent.py` script. For example:
```sh
python agents/default_comparer_agent.py
```

This module is designed to calculate accuracy using a default comparison method. Additionally, it includes a function that provides a detailed breakdown of accuracy for each dataset. You have the option to specify the file paths for the responses and the answers if necessary.

## Running Default Comparer on All Responses

The `main.py` file is designed to run the default comparer on all model responses and print the results to the terminal. To execute it, simply run:

```sh
python main.py
```

This script will load all response files, compare them to the actual answers using the default comparison method from databench-eval, and output the accuracy results directly to the terminal as a breakdown for each dataset aswell as the overall score.

## Results

The results of the model evaluations are documented in two files:

- `Accuracy_Results.md`: This file provides a summary of the accuracy results for different models and techniques on sample data. It includes the overall accuracy for each method such as Zero-shot baseline, Zero-shot ICL, CoT, Code-based prompting, and Prompt-Engineering for both GPT-3.5-turbo and 4o-mini models.

- `Dataset_Accuracy.md`: This file provides a detailed breakdown of the accuracy for each dataset and model. It includes the accuracy for each dataset and the overall accuracy for each model response file. This allows for a more granular analysis of the model performance across different datasets.

## Data Visualization
To visualize the results of the model evaluations, you can use the `dataset_viz.py` script. This script generates several visualizations to help analyze the performance of different models across various datasets.

### Generating Visualizations

1. Ensure you have the necessary dependencies installed:
  ```sh
  pip install pandas matplotlib numpy
  ```

2. Run the `dataset_viz.py` script:
  ```sh
  python utilities/dataset_viz.py
  ```

This will create and save the following visualizations as PNG files:

- `stacked_by_dataset.png`: A stacked bar graph showing the cumulative accuracy of each model across different datasets.
- `stacked_by_model.png`: A stacked bar graph showing the cumulative accuracy of each dataset across different models.
- `overall_accuracy.png`: A bar graph comparing the overall accuracy of each model.

These visualizations provide a clear and concise way to compare model performance and identify the best-performing models and datasets.


## Updating Results
Whenever new results is obtained and push to the main repo, please update the following files:
- `Accuracy_Results.md`
- `Dataset_Accuracy.md`
- `Type_Accuracy.md`
- `model_accuracy_comparison.csv`
- `type_accuracy.csv`
- Run the `dataset_viz.py` file for updated .png files
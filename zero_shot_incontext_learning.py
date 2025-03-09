import openai
import os
from dataAgent import DataAgent  # Import the DataAgent class

api_key = os.getenv("OPENAI_API_KEY")  

openai.api_key = api_key

# Initialize DataAgent and load data from competition directory
competition_directory = os.path.join(os.path.dirname(__file__), "competition")
agent = DataAgent()
agent.load_data(competition_directory)

def query_gpt_icl(csv_data, question):
    """
    Queries OpenAI's GPT model using the given tabular data and question.
    """
    prompt = f"""
    You are an AI answering questions based on tabular data.

    Here is the dataset:
    ```
    {csv_data}
    ```
    Please answer the following question in JSON format:
    Question: {question}

    Example response:
    {{
        "answer": "<your answer>",
        "columns_used": ["<column1>", "<column2>"],
        "explanation": "<brief reasoning>"
    }}
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use "gpt-4" if available
        messages=[
            {"role": "system", "content": "You are a data analyst answering questions about tabular data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response["choices"][0]["message"]["content"]

def get_csv_data(dataset_name, dataset_type="sample"):
    """
    Retrieve CSV data as a string using DataAgent.
    
    Args:
        dataset_name (str): The competition dataset folder name.
        dataset_type (str): Either 'sample' or 'all' (default: 'sample').

    Returns:
        str: CSV content as a string.
    """
    if dataset_name in agent.data and dataset_type in agent.data[dataset_name]:
        csv_data = agent.data[dataset_name][dataset_type]  # Retrieve data list
        return "\n".join([",".join(row) for row in csv_data])  # Convert to CSV string
    else:
        raise FileNotFoundError(f"Dataset {dataset_name}/{dataset_type}.csv not found.")

# Example usage
dataset_name = "071_COL"  # Example dataset
dataset_type = "sample"  # Can also be "all"
csv_data = get_csv_data(dataset_name, dataset_type)

question = "What is the most expensive city in this dataset?"
print(query_gpt_icl(csv_data, question))
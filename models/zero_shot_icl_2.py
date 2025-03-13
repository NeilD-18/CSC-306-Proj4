import os
from openai import OpenAI
import json  # Ensure json module is imported

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Add the parent directory to the system path

from agents.dataAgent import DataAgent  # Import the DataAgent class
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

api_key = os.getenv("OPENAI_API_KEY")

class ZeroShotModelICL2:
    def __init__(self, api_key=None, competition_directory=None, data=None):
        """
        Initialize the ZeroShotModel with OpenAI API key and data directory.
        
        Args:
            api_key (str, optional): OpenAI API key. Defaults to environment variable.
            competition_directory (str, optional): Path to competition data directory.
        """
        # Set up API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # Check if API key is available
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Either pass it as api_key parameter or "
                "set the OPENAI_API_KEY environment variable."
            )

        # Initialize OpenAI client with the API key

        # Set up DataAgent
        self.agent = DataAgent()
        if not data:
            self.competition_directory = competition_directory or os.path.join(os.path.dirname(__file__), "../competition")
            self.agent.load_data(self.competition_directory)
        else:
            self.agent.data = data
        self.client = OpenAI(api_key=self.api_key)

    def query_gpt_icl(self, csv_data, question):
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
        Make sure the answer you provide is simple and either of the following data-type:
        - String
        - Integer
        - Float
        - List of strings
        - List of integers
        - List of floats

        Example response:
        {{
            "answer": "<your answer>",
            "columns_used": ["<column1>", "<column2>"],
            "explanation": "<brief reasoning>"
        }}
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analyst answering questions about tabular data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0
        )

        return response.choices[0].message.content.strip()

    def get_csv_data(self, dataset_name, dataset_type="sample"):
        """
        Retrieve CSV data as a string using DataAgent.
        
        Args:
            dataset_name (str): The competition dataset folder name.
            dataset_type (str): Either 'sample' or 'all' (default: 'sample').

        Returns:
            str: CSV content as a string.
        """
        if dataset_name in self.agent.data and dataset_type in self.agent.data[dataset_name]:
            csv_data = self.agent.data[dataset_name][dataset_type]  # Retrieve data list
            return "\n".join([",".join(row) for row in csv_data])  # Convert to CSV string
        else:
            raise FileNotFoundError(f"Dataset {dataset_name}/{dataset_type}.csv not found.")

    def ask_question(self, dataset_name, question, dataset_type="sample"):
        """
        Convenience method to ask a question about a specific dataset.
        
        Args:
            dataset_name (str): The competition dataset folder name.
            question (str): The question to ask about the dataset.
            dataset_type (str): Either 'sample' or 'all' (default: 'sample').
            
        Returns:
            str: The model's response.
        """
        csv_data = self.get_csv_data(dataset_name, dataset_type)
        return self.query_gpt_icl(csv_data, question)


# Example usage
if __name__ == "__main__":
    # Initialize the model
    model = ZeroShotModelICL2()

    # Ask a question about a dataset
    dataset_name = "071_COL"
    question = "What is the most expensive city in this dataset?"

    response = model.ask_question(dataset_name, question)
    # print(response)
    try:
        response_json = json.loads(response)
        print(response_json["answer"])
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON response: {e}")
        print(f"Response: {response}")
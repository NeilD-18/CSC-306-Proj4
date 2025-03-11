import sys
import os
from openai import OpenAI

import dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.dataAgent import DataAgent  # Import the DataAgent class

dotenv.load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("OPENAI_API_KEY")

class ZeroShotModel:
    def __init__(self, api_key=None, competition_directory=None):
        """
        Initialize the Zero-Shot Baseline Model with OpenAI API key and data directory.
        
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
        self.client = OpenAI(api_key=self.api_key)

        # Set up DataAgent
        self.agent = DataAgent()
        self.competition_directory = competition_directory or os.path.join(os.path.dirname(__file__), "../competition")
        self.agent.load_data(self.competition_directory)

    def query_gpt_baseline(self, csv_data, question):
        """
        Queries OpenAI's GPT model using a simple, direct prompt with the given tabular data and question.
        """
        prompt = f"""
        You are an AI answering questions based on tabular data.

        Here is the dataset:
        ```
        {csv_data}
        ```

        Answer the following question directly, without explanation:
        {question}

        Example response:
        {{
            "answer": "<your answer>",
            "columns_used": ["<column1>", "<column2>"],
            "explanation": "<brief reasoning>"
        }}
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use "gpt-4" if available
            messages=[
                {"role": "system", "content": "You are a data analyst answering questions about tabular data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
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
        Ask a question about a dataset using the simple baseline approach.
        
        Args:
            dataset_name (str): The competition dataset folder name.
            question (str): The question to ask about the dataset.
            dataset_type (str): Either 'sample' or 'all' (default: 'sample').
            
        Returns:
            str: The model's response.
        """
        csv_data = self.get_csv_data(dataset_name, dataset_type)
        return self.query_gpt_baseline(csv_data, question)


# Example usage
if __name__ == "__main__":
    
    # Initialize the baseline model
    model = ZeroShotModel()

    # Ask a question about a dataset
    dataset_name = "071_COL"
    question = "What is the most expensive city in this dataset?"

    response = model.ask_question(dataset_name, question)
    print(response)

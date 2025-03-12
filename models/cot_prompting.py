import os
from openai import OpenAI
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.dataAgent import DataAgent  # Import the DataAgent class

class CoTPromptingModel:
    def __init__(self, api_key=None, competition_directory=None):
        """
        Initialize the Chain of Thought Prompting Model with OpenAI API key and data directory.
        
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

    def identify_relevant_columns(self, csv_data, question):
        """
        Step 1: Queries GPT-3.5 to determine which columns in the dataset are relevant for answering the question.
        """
        prompt = f"""
        You are analyzing a dataset and determining which columns are most relevant for answering a question.

        Here is the dataset:
        ```
        {csv_data}
        ```

        Identify the column names that are necessary to answer this question:
        "{question}"

        Respond with a list of column names in JSON format:
        {{
            "columns_used": ["<column1>", "<column2>", ...]
        }}
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use "gpt-4" if available
            messages=[
                {"role": "system", "content": "You are a data analyst identifying important columns in a dataset."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0
        )

        # Extract columns from response (expects JSON format)
        import json
        try:
            columns_info = json.loads(response.choices[0].message.content.strip())
            return columns_info.get("columns_used", [])
        except json.JSONDecodeError:
            return []

    def query_gpt_chain_of_thought(self, csv_data, relevant_columns, question):
        """
        Step 2 & 3: Queries GPT-3.5 with Chain of Thought reasoning to systematically answer the question.
        """
        prompt = f"""
        You are an AI answering questions based on tabular data.

        Here is the dataset:
        ```
        {csv_data}
        ```

        The most relevant columns for answering the question are: {', '.join(relevant_columns)}.

        Step 1: First, analyze the values in these columns and explain how they can be used to answer the question.

        Step 2: Based on this analysis, derive the final answer.

        Now, answer the question:
        "{question}"
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
                {"role": "system", "content": "You are a data analyst reasoning through tabular data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
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
        Ask a question about a dataset using the Chain of Thought prompting approach.
        
        Args:
            dataset_name (str): The competition dataset folder name.
            question (str): The question to ask about the dataset.
            dataset_type (str): Either 'sample' or 'all' (default: 'sample').
            
        Returns:
            str: The model's response.
        """
        # Load dataset as a string
        csv_data = self.get_csv_data(dataset_name, dataset_type)

        # Step 1: Identify relevant columns
        relevant_columns = self.identify_relevant_columns(csv_data, question)
        print(f"Identified Relevant Columns: {relevant_columns}")

        # Step 2 & 3: Use CoT prompting to answer the question
        return self.query_gpt_chain_of_thought(csv_data, relevant_columns, question)


# Example usage
if __name__ == "__main__":
    # Initialize the CoT model
    model = CoTPromptingModel()

    # Ask a question about a dataset
    dataset_name = "071_COL"
    question = "What is the most expensive city in this dataset?"

    response = model.ask_question(dataset_name, question)
    print(response)

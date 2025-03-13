import os
import sys
import json
import re
import csv
import pandas as pd
import numpy as np
from io import StringIO
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.dataAgent import DataAgent  # Import the DataAgent class

class CodeBasedModel:
    def __init__(self, api_key=None, competition_directory=None, data=None):
        """
        Initialize the Code-Based Model with OpenAI API key and data directory.
        
        Args:
            api_key (str, optional): OpenAI API key. Defaults to environment variable.
            competition_directory (str, optional): Path to competition data directory.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.agent = DataAgent()
        if not data:
            self.competition_directory = competition_directory or os.path.join(os.path.dirname(__file__), "../competition")
            self.agent.load_data(self.competition_directory)
        else:
            self.agent.data = data
        self.client = OpenAI(api_key=self.api_key)

    def query_gpt_code(self, csv_data, column_names, question):
        """
        Queries OpenAI's GPT model to generate Python code for answering the question.
        
        Args:
            csv_data (str): CSV data as a string.
            column_names (list): List of column names.
            question (str): The question to answer.
        
        Returns:
            str: Generated Python code.
        """
        # Safely escape CSV data as JSON
        safe_csv_data = json.dumps(csv_data)
        
        prompt = f"""
        You are an AI assistant that generates Python code to answer questions based on a tabular dataset.
        
        Below is a dataset stored in a pandas DataFrame:
        ```python
        import pandas as pd
        from io import StringIO
        
        data = {safe_csv_data}
        df = pd.read_csv(StringIO(data))
        ```
        
        The dataset contains the following columns: {', '.join(column_names)}
        
        ### Task:
        Write a **Python function** called `answer(df)` that computes the answer to the following question:
        **Question:** {question}
        
        The function should return a Python dictionary in the following format:
        ```python
        {{
            "answer": "<your answer>",
            "columns_used": ["<column1>", "<column2>"],
            "explanation": "<brief reasoning>"
        }}
        ```
        
        Ensure:
        - The function uses only the necessary columns.
        - The output is a Python dictionary (not a JSON string).
        - The answer is one of the following data types: String, Integer, Float, List of Strings, List of Integers, or List of Floats.
        
        ONLY RETURN THE PYTHON CODE, DO NOT RETURN ANYTHING ELSE.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analyst answering questions about tabular data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0
        )
        
        return response.choices[0].message.content.strip()
    
    def __safe_convert_numeric(self, df):
        """
        Converts only columns that should be numeric, leaving text-based columns unchanged.
        
        Uses an 80% numeric heuristic.
        
        Args:
            df (pd.DataFrame): The DataFrame to process.
        
        Returns:
            pd.DataFrame: A DataFrame with properly converted numeric values.
        """
        for col in df.columns:
            # Convert column to numeric with coercion; count how many values are numeric
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            num_numeric = numeric_series.notna().sum()
            total = len(df[col])
            if total > 0 and (num_numeric / total) >= 0.8:
                df[col] = numeric_series
        return df
    
    def get_csv_data(self, dataset_name, dataset_type="sample"):
        """
        Retrieve CSV data as a string using DataAgent.
        
        Uses the csv module to properly quote values.
        
        Args:
            dataset_name (str): The competition dataset folder name.
            dataset_type (str): Either 'sample' or 'all'.
        
        Returns:
            str: CSV content as a string.
        """
        if dataset_name in self.agent.data and dataset_type in self.agent.data[dataset_name]:
            data_list = self.agent.data[dataset_name][dataset_type]  # list of lists
            output = StringIO()
            writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
            writer.writerows(data_list)
            return output.getvalue().strip()
        else:
            raise FileNotFoundError(f"Dataset {dataset_name}/{dataset_type}.csv not found.")
    
    def execute_generated_code(self, code, df):
        """
        Executes the generated Python code and returns the computed answer in JSON format.
        
        Steps:
          1. Remove markdown artifacts.
          2. Extract only the function definition.
          3. Build an execution scope with pd and np.
          4. Convert only numeric columns.
          5. Fill missing values (numeric: median, text: empty string).
          6. Exec the function and return its output.
        
        Args:
            code (str): The Python code generated by the LLM.
            df (pd.DataFrame): The dataset as a DataFrame.
        
        Returns:
            str: A JSON-formatted string with the result or an error.
        """
        try:
            # Remove markdown backticks (opening and closing)
            code = re.sub(r"```(?:python)?\n?", "", code)
            code = re.sub(r"\n?```", "", code)
            
            # Extract only the function definition that starts with "def answer(df):"
            code_match = re.search(r"(def answer\(df\):.*)", code, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            else:
                return json.dumps({"error": "Could not extract the function definition."}, indent=4)
            
            # Define the local scope for execution
            local_scope = {"df": df, "pd": pd, "np": np}
            
            # Convert columns that should be numeric
            df = self.__safe_convert_numeric(df)
            
            # Fill missing values based on dtype:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Use median if possible; if no numeric values, fill with 0
                    median_val = df[col].median() if not df[col].empty else 0
                    df[col] = df[col].fillna(median_val)
                else:
                    df[col] = df[col].fillna("")
            
            # Execute the cleaned function code
            exec(code, {}, local_scope)
            
            if "answer" in local_scope:
                result = local_scope["answer"](df)
                if isinstance(result, dict):
                    return json.dumps(result, indent=4, default=str)
                else:
                    return json.dumps({"error": "Generated function did not return a dictionary."}, indent=4)
            return json.dumps({"error": "Function `answer(df)` was not defined in the generated code."}, indent=4)
        
        except Exception as e:
            print("\n--- Extracted Code Before Execution ---\n", code)
            return json.dumps({"error": f"Error executing code: {str(e)}"}, indent=4)
    
    def ask_question(self, dataset_name, question, dataset_type="sample"):
        """
        Asks a question about a dataset and returns the answer.
        
        Args:
            dataset_name (str): The competition dataset folder name.
            question (str): The question to ask.
            dataset_type (str, optional): Either 'sample' or 'all'.
        
        Returns:
            str: A JSON string with the result or error.
        """
        try:
            csv_data = self.get_csv_data(dataset_name, dataset_type)
            # Use on_bad_lines="skip" for compatibility with pandas 1.3+
            df = pd.read_csv(StringIO(csv_data), on_bad_lines="skip")
            column_names = df.columns.tolist()
            generated_code = self.query_gpt_code(csv_data, column_names, question)
            return self.execute_generated_code(generated_code, df)
        except Exception as e:
            return json.dumps({"error": f"Error loading CSV: {str(e)}"}, indent=4)

# Example usage
if __name__ == "__main__":
    model = CodeBasedModel()
    dataset_name = "071_COL"
    question = "What is the most expensive city in this dataset?"
    response = model.ask_question(dataset_name, question)
    print(response)

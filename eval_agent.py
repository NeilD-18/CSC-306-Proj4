from databench_eval import Runner, Evaluator
from databench_eval.utils import load_qa
from models.zero_shot_incontext_learning import ZeroShotModelICL
from models.cot_prompting import CoTPromptingModel
from models.zero_shot_icl_2 import ZeroShotModelICL2
from models.zero_shot_baseline import ZeroShotModel

import csv
import os
import json
from tqdm import tqdm
from models.zero_shot_baseline import ZeroShotModel
from models.code_based_learning import CodeBasedModel
from models.prompt_engineering import PromptEngineering

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


class EvalAgent:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = ZeroShotModelICL(api_key=self.api_key)
    
    def model_call(self, prompts: list[str], model=None) -> list[str]:
        """ 
        Call model on a batch of prompts.
        
        Args:
            prompts: List of prompts to process
            model: Model instance with ask_question method (uses self.model if None)
        """
        if model is None:
            model = self.model
            
        responses = []
        for prompt in prompts:
            question = prompt['question']
            dataset = prompt['dataset']
            response = model.ask_question(dataset, question)
            try:
                response_json = json.loads(response)
                print(response_json)
                 # ✅ Check if "answer" exists before accessing it
                if "answer" in response_json:
                    responses.append(response_json["answer"])
                else:
                    responses.append("ERROR") 
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON response: {e}")
                responses.append(f"Failed to decode JSON response: {e}")
        return responses

    def load_test_qa(self, filepath='competition/test_qa.csv'):
        """
        Load test questions and datasets from a CSV file.
        
        Args:
            filepath: Path to the CSV file containing questions and datasets
            
        Returns:
            A list of dictionaries with 'question' and 'dataset' keys
        """
        qa_data = []
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(current_dir, filepath)
        
        with open(full_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                qa_data.append({
                    'question': row['question'],
                    'dataset': row['dataset']
                })
        
        return qa_data

    def run_batch(self, prompts, batch_size=10, model=None):
        """
        Run the model on batches of prompts without using Runner.
        
        Args:
            prompts: List of prompts to process
            batch_size: Size of batches to process
            model: Model to use (uses self.model if None)
            
        Returns:
            List of model responses
        """
        responses = []
        
        # Process prompts in batches
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = self.model_call(batch_prompts, model)
            responses.extend(batch_responses)
            
        return responses
        
    def evaluate(self, test_qa_path='competition/test_qa.csv', save_path="responses.txt", model=None):
        """
        Run evaluation on the test dataset and print metrics
        
        Args:
            test_qa_path: Path to the test QA file
            save_path: Path to save responses
            model: Model to use (uses self.model if None)
        """
        print("Loading test data...")
        test_qa = self.load_test_qa(test_qa_path)

        from datasets import Dataset
        import pandas as pd
        
        # Convert test_qa to a Dataset object
        df = pd.DataFrame(test_qa)
        qa_dataset = Dataset.from_pandas(df)
        
        # Create a wrapper function that shows progress
        def model_call_with_progress(prompts):
            print(f"Processing batch of {len(prompts)} prompts...")
            return self.model_call(prompts, model)
        
        print("Running evaluation...")
        # Now use Runner with this dataset
        responses = Runner(model_call_with_progress, qa=qa_dataset).run(test_qa, save=save_path)

        print("Reading responses...")
        # Read the responses from the saved file with tqdm
        responses = []
        with open(save_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in tqdm(lines, desc="Processing responses"):
                responses.append(line.strip())

        print("Calculating metrics...")
        acc = Evaluator().eval(responses)
        acc_lite = Evaluator().eval(responses, lite=True)

        print(f"Accuracy: {acc}")
        print(f"Lite accuracy: {acc_lite}")
        print(f"Responses saved to {save_path}")
        
        return acc, acc_lite, responses


# Example usage:
if __name__ == "__main__":
    agent = EvalAgent()
    api_key = os.getenv("OPENAI_API_KEY")

    cot = CoTPromptingModel(api_key=api_key)
    icl = ZeroShotModelICL2(api_key=api_key)
    baseline = ZeroShotModel(api_key=api_key)
    pe = PromptEngineering(api_key=api_key)
    cbl = CodeBasedModel(api_key=api_key)
    
    #agent.evaluate(save_path="responses_cbl_4o-mini.txt", test_qa_path='competition/test_qa.csv',model=cbl)
    agent.evaluate(save_path="responses_Cot_3.5-turbou.txt", test_qa_path='competition/test_qa.csv',model=cot)
    # agent.evaluate(save_path="responses_zero_shot_icl_4o-mini.txt", test_qa_path='competition/test_qa.csv',model=icl)
    # agent.evaluate(save_path="responses_zero_shot_baseline_4o-mini.txt", test_qa_path='competition/test_qa.csv',model=baseline)
    #agent.evaluate(save_path="responses_pe_4o-mini.txt", test_qa_path='competition/test_qa.csv',model=pe)
    
    # Alternative batch processing approach:
    # test_qa = agent.load_test_qa()
    # responses = agent.run_batch(test_qa, batch_size=10)
    # print(responses)
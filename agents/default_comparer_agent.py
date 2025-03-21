import os
import sys
from databench_eval import Evaluator
"""
This module provides a DefaultComparer class that compares model responses to actual answers
using a default comparison method provided by the Evaluator class from the databench_eval module.
Classes:
    DefaultComparer: A class to compare responses from a file to actual answers and calculate accuracy.
Methods:
    __init__(self, file_path):
        Initializes the DefaultComparer with the given file path to the responses file.
    default_accuracy(self):
        Reads the responses, answers, and semantics from their respective files.
        Uses the default_compare method of the Evaluator class to compare each response with the corresponding answer and semantic.
        Calculates and prints the accuracy of the responses.
Usage:
    The script can be run directly, and it will instantiate a DefaultComparer with a specified file path to the responses file.
    It will then call the default_accuracy method to perform the comparison and print the accuracy.
"""


class DefaultComparer:
    def __init__(self):
        self.evaluator = Evaluator()
        self.file_path = None
        self.answers_lite = "answers/answers_lite.txt"
        self.answers_all = "answers/answers.txt"
        self.semantics = "answers/semantics.txt"
        self.answers = None

        self.answer_indices = {
        66: (0, 40),
        67: (40, 69),
        68: (69, 103),
        69: (103, 138),
        70: (138, 167),
        71: (167, 203),
        72: (203, 242),
        73: (242, 274),
        74: (274, 309),
        75: (309, 338),
        76: (338, 374),
        77: (374, 405),
        78: (405, 444),
        79: (444, 482),
        80: (482, 524)
    }
    

    def semantic_accuracy(self, filename):
        """
        Evaluate model performance based on semantic types.
        
        Parameters:
            filename (str): Path to the response file to evaluate
            
        Returns:
            dict: Dictionary containing accuracy metrics for each semantic type
        """
        try:
            # Read the files
            with open(filename, "r", encoding='utf-8') as f:
                responses = f.read().splitlines()
            with open(self.answers_lite, "r", encoding='utf-8') as f:
                answers = f.read().splitlines()
            with open(self.semantics, "r", encoding='utf-8') as f:
                semantics = f.read().splitlines()

            # Initialize counters for each semantic type
            semantic_stats = {
                'boolean': {'correct': 0, 'total': 0},
                'category': {'correct': 0, 'total': 0},
                'number': {'correct': 0, 'total': 0},
                'list[category]': {'correct': 0, 'total': 0},
                'list[number]': {'correct': 0, 'total': 0}
            }

            # Evaluate each response
            for response, answer, semantic in zip(responses, answers, semantics):
                semantic = semantic.strip()
                if semantic not in semantic_stats:
                    continue
                    
                semantic_stats[semantic]['total'] += 1
                if self.evaluator.default_compare(response, answer, semantic):
                    semantic_stats[semantic]['correct'] += 1

            # Print results
            print(f"\nSemantic Type Analysis for: {os.path.basename(filename)}")
            print("-" * 50)
            print("Type            Correct     Total    Accuracy")
            print("-" * 50)

            for semantic_type, stats in semantic_stats.items():
                if stats['total'] > 0:
                    accuracy = stats['correct'] / stats['total']
                    print(f"{semantic_type:<15} {stats['correct']:<10} {stats['total']:<9} {accuracy:.2f}")
                else:
                    print(f"{semantic_type:<15} No examples found")

            # Calculate overall accuracy
            total_correct = sum(stats['correct'] for stats in semantic_stats.values())
            total_samples = sum(stats['total'] for stats in semantic_stats.values())
            overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

            print("-" * 50)
            print(f"Overall Accuracy: {overall_accuracy:.2f}")
            
            return semantic_stats

        except Exception as e:
            print(f"Error in semantic analysis: {str(e)}")
            return None
    
    def default_accuracy(self,file_path, answers=None):
        """
        Calculate the accuracy of responses compared to the provided answers using a default comparison method.
        Parameters:
        answers (str, optional): Path to the file containing the correct answers. If not provided, self.answers_lite will be used.
        Reads the following files:
        - self.file_path: File containing the responses to be evaluated.
        - answers: File containing the correct answers.
        - self.semantics: File containing the semantics for comparison.
        Uses the default_compare method of the evaluator to compare each response with the corresponding answer and semantic.
        Prints the accuracy as a percentage.
        Returns:
        None
        """
        self.file_path = file_path
        if not answers:
            self.answers = self.answers_lite
        else:
            self.answers = answers
        # Read the files
        with open(self.file_path, "r") as f:
            responses = f.read().splitlines()
        with open(self.answers, "r") as f:
            answers = f.read().splitlines()
        with open(self.semantics, "r") as f:
            semantics = f.read().splitlines()
        
        # Use default_compare method
        correct = 0
        for response, answer, semantic in zip(responses, answers, semantics):
            if self.evaluator.default_compare(response, answer, semantic):
                correct += 1
        
        accuracy = correct / len(answers)
        print(f"Accuracy: {accuracy:.2f}")

    def dataset_accuracy(self, filename):
        # Read the files
        with open(filename, "r") as f:
            responses = f.read().splitlines()
        with open(self.answers_lite, "r") as f:
            answers = f.read().splitlines()
        with open(self.semantics, "r") as f:
            semantics = f.read().splitlines()
        
        print(f"Breakdown for model: {filename}")
        # Calculate accuracy for each dataset
        print("Accuracy by dataset:")
        print("-" * 30)
        
        for dataset_num, (start_idx, end_idx) in self.answer_indices.items():
            # Get slices for current dataset
            dataset_responses = responses[start_idx:end_idx]
            dataset_answers = answers[start_idx:end_idx]
            dataset_semantics = semantics[start_idx:end_idx]
            
            # Calculate accuracy for current dataset
            correct = 0
            for response, answer, semantic in zip(dataset_responses, dataset_answers, dataset_semantics):
                if self.evaluator.default_compare(response, answer, semantic):
                    correct += 1
            
            dataset_accuracy = correct / (end_idx - start_idx)
            print(f"Dataset {dataset_num}: {dataset_accuracy:.2f}")
        
        # Calculate overall accuracy
        total_correct = 0
        for response, answer, semantic in zip(responses, answers, semantics):
            if self.evaluator.default_compare(response, answer, semantic):
                total_correct += 1
        
        overall_accuracy = total_correct / len(answers)
        print("-" * 30)
        print(f"Overall Accuracy: {overall_accuracy:.2f}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = 'model_responses/responses_cbl_4o-mini33percent.txt'
    comparer = DefaultComparer()
    # comparer.default_accuracy(file_path=filepath)
    # comparer.dataset_accuracy(filepath)
    comparer.semantic_accuracy(filepath)
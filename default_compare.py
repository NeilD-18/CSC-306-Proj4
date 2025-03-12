from databench_eval import Evaluator

if __name__ == "__main__":
    # Create an instance of Evaluator
    evaluator = Evaluator()
    
    # Read the files
    with open("responses_Cot_3.5-turbo.txt", "r") as f:
        responses = f.read().splitlines()
    with open("answers/answers.txt", "r") as f:
        answers = f.read().splitlines()
    with open("answers/semantics.txt", "r") as f:
        semantics = f.read().splitlines()
    
    # Use default_compare method
    correct = 0
    for response, answer, semantic in zip(responses, answers, semantics):
        if evaluator.default_compare(response, answer, semantic):
            correct += 1
    
    accuracy = correct / len(answers)
    print(f"Accuracy: {accuracy:.2f}")
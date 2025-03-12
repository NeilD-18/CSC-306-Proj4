from databench_eval import Evaluator
# with open("answers/answers.txt", "r") as f:
#     answers = f.read().splitlines()
#     answer_66 = answers[:40]
#     answer_67 = answers[40:69]
#     answer_68 = answers[69:103]
#     answer_69 = answers[103:138]
#     answer_70 = answers[138:167]
#     answer_71 = answers[167:203]
#     answer_72 = answers[203:242]
#     answer_73 = answers[242:274]
#     answer_74 = answers[274:309]
#     answer_75 = answers[309:338]
#     answer_76 = answers[338:374]
#     answer_77 = answers[374:405]
#     answer_78 = answers[405:444]
#     answer_79 = answers[444:482]
#     answer_80 = answers[482:524]


if __name__ == "__main__":
    # Create an instance of Evaluator
    evaluator = Evaluator()
    answer_indices = {
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
    filename = "model_responses/responses_cbl_4o-mini33percent.txt"
    # Read the files
    with open(filename, "r") as f:
        responses = f.read().splitlines()
    with open("answers/answers.txt", "r") as f:
        answers = f.read().splitlines()
    with open("answers/semantics.txt", "r") as f:
        semantics = f.read().splitlines()
    
    print(f"Breakdown for model: {filename} ")
    # Calculate accuracy for each dataset
    print("Accuracy by dataset:")
    print("-" * 30)
    
    for dataset_num, (start_idx, end_idx) in answer_indices.items():
        # Get slices for current dataset
        dataset_responses = responses[start_idx:end_idx]
        dataset_answers = answers[start_idx:end_idx]
        dataset_semantics = semantics[start_idx:end_idx]
        
        # Calculate accuracy for current dataset
        correct = 0
        for response, answer, semantic in zip(dataset_responses, dataset_answers, dataset_semantics):
            if evaluator.default_compare(response, answer, semantic):
                correct += 1
        
        dataset_accuracy = correct / (end_idx - start_idx)
        print(f"Dataset {dataset_num}: {dataset_accuracy:.2f}")
    
    # Calculate overall accuracy
    total_correct = 0
    for response, answer, semantic in zip(responses, answers, semantics):
        if evaluator.default_compare(response, answer, semantic):
            total_correct += 1
    
    overall_accuracy = total_correct / len(answers)
    print("-" * 30)
    print(f"Overall Accuracy: {overall_accuracy:.2f}")
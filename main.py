from agents.default_comparer_agent import DefaultComparer
import os
import sys


def main():
    zs_icl = "model_responses/responses_zero_shot_icl.txt"
    zs_cot = "model_responses/responses_zero_shot_CoT.txt"
    zs_base_4o_mini = "model_responses/responses_zero_shot_baseline_4o-mini.txt"
    zs_base_3_5_turbo = "model_responses/responses_zero_shot_baseline_3.5-turbo.txt"
    pe = "model_responses/responses_PE.txt"
    pe_4o_mini = "model_responses/responses_pe_4o-mini.txt"
    icl_4o_mini = "model_responses/responses_icl_4o-mini.txt"
    cot_4o_mini = "model_responses/responses_Cot_4o-mini.txt"
    cot_3_5_turbo = "model_responses/responses_Cot_3.5-turbo.txt"
    cbl_4o_mini_33 = "model_responses/responses_cbl_4o-mini33percent.txt"
    cbl_4o_mini_32 = "model_responses/responses_cbl_4o-mini32percent.txt"

    comparer = DefaultComparer()
    comparer.dataset_accuracy(zs_icl)
    comparer.dataset_accuracy(zs_cot)
    comparer.dataset_accuracy(zs_base_4o_mini)
    comparer.dataset_accuracy(zs_base_3_5_turbo)
    comparer.dataset_accuracy(pe)
    comparer.dataset_accuracy(pe_4o_mini)
    comparer.dataset_accuracy(icl_4o_mini)
    comparer.dataset_accuracy(cot_4o_mini)
    comparer.dataset_accuracy(cot_3_5_turbo)
    comparer.dataset_accuracy(cbl_4o_mini_33)
    comparer.dataset_accuracy(cbl_4o_mini_32)

if __name__ == "__main__":
    main()
"""Postprocess for gpt outputs."""

import re
from typing import List, Dict, Optional
import json
import argparse


def parse_fact_enhance_classify_res(raw_response: str):
    replace_map = {
        "#Prediction:": "Prediction:",
        "#Search Query:": "Search Query:",
        "#Command:": "Command:"
    }
    for old_text, new_text in replace_map.items():
        raw_response = raw_response.replace(old_text, new_text)
    true_resp = re.split("Command:", raw_response)[0]
    split_response = re.split("Prediction:|Search Query:", true_resp)
    if len(split_response) < 2:
        # print(f"Illegal respnse :{raw_response}")
        return None
    analysis, final_prediction = split_response[0], split_response[1]
    final_prediction = final_prediction.strip()
    if final_prediction not in ["<need>", "<no need>"]:
        # print(f"Illegal final_prediction :{raw_response}")
        return None
    queris = ""
    if final_prediction == "<need>":
        if len(split_response) == 3:
            queris = split_response[2]
    analysis = analysis.strip()
    queris = queris.strip()
    return {"analysis": analysis, "final_prediction": final_prediction, "queris": queris, }


def parse_fact_generation_res(raw_response: str):
    return {"knowledge": raw_response.strip()}


def parse_test_generation_res(raw_response: str):
    def veryfy_test(test_sample: Dict[str, str]) -> Optional[Dict[str, str]]:
        options = re.split("\(A\)|\(B\)|\(C\)|\(D\)", test_sample["options"])
        choices = [option.strip() for option in options if option]
        if len(choices) != 4:
            return None
        norm_ans = "".join(c for c in test_sample["answer"].upper() if c in ["A", "B", "C", "D"])
        if len(norm_ans) != 1:
            return None
        test_sample["options"], test_sample["answer"] = choices, norm_ans
        return test_sample
    replace_map = {
        "#Question:": "Question:",
        "#Options:": "Options:",
        "#Analysis:": "Analysis:",
        "#Answer:": "Answer:",
    }
    for old_text, new_text in replace_map.items():
        raw_response = raw_response.replace(old_text, new_text)
    questions = re.split("Question:", raw_response)
    q_id = 0
    res_questions = []
    for single_question in questions:
        question_components = re.split("Options:|Analysis:|Answer:", single_question)
        if len(question_components) != 4:
            continue
        parse_question = {
            "question": question_components[0].strip(),
            "options": question_components[1].strip(),
            "analysis": question_components[2].strip(),
            "answer": question_components[3].strip().replace("-", ""),
        }
        normalized_test = veryfy_test(parse_question)
        q_id += 1
        res_questions.append(normalized_test)
    return {"tests": res_questions}


def parse_test_generation(
    raw_responses: List[str],
    subset_name: str = "sft",
) -> List[Dict[str, str]]:
    if isinstance(raw_responses[0], dict):  # fix api res as input
        raw_responses = [exp["raw_response"] for exp in raw_responses]
    def veryfy_test(test_sample: Dict[str, str]) -> Optional[Dict[str, str]]:
        options = re.split("\(A\)|\(B\)|\(C\)|\(D\)", test_sample["options"])
        choices = [option.strip() for option in options if option]
        if len(choices) != 4:
            return None
        norm_ans = "".join(c for c in test_sample["answer"].upper() if c in ["A", "B", "C", "D"])
        if len(norm_ans) != 1:
            return None
        test_sample["options"], test_sample["answer"] = choices, norm_ans
        return test_sample
    replace_map = {
        "#Question:": "Question:",
        "#Options:": "Options:",
        "#Analysis:": "Analysis:",
        "#Answer:": "Answer:",
    }
    fail_q_num, res_questions = 0, []
    for idx, raw_response in enumerate(raw_responses):
        instance_idx = f"idx_{subset_name}{idx}"
        for old_text, new_text in replace_map.items():
            raw_response = raw_response.replace(old_text, new_text)
        questions = re.split("Question:", raw_response)
        q_id = 0
        for single_question in questions:
            question_components = re.split("Options:|Analysis:|Answer:", single_question)
            if len(question_components) != 4:
                fail_q_num += 1
                continue
            parse_question = {
                "question": question_components[0].strip(),
                "options": question_components[1].strip(),
                "analysis": question_components[2].strip(),
                "answer": question_components[3].strip().replace("-", ""),
            }
            normalized_test = veryfy_test(parse_question)
            if not normalized_test:
                fail_q_num += 1
                continue
            parse_question["idx"] = f"{instance_idx}_test{q_id}"
            res_questions.append(parse_question)
            q_id += 1
    return res_questions


def fact_enhance_classify_post_process(input_file, output_file, mode):
    """
    postprocess for fact enhance classify results
    1. parse result;
    2. select result with need fact for fact gen;
    """
    data = [json.loads(line.strip()) for line in open(input_file, "r")]
    processed_data = []
    for example in data:
        if mode == "parse_res":
            processed_example = parse_fact_enhance_classify_res(example["raw_response"])
            if processed_example is not None:
                example.update(processed_example)
            processed_data.append(example)
        elif mode == "select_need":
            if "final_prediction" in example and example["final_prediction"] == "<need>":
                processed_example = {"input": example["original_input"]["input"],
                                     "output": example["original_input"]["output"],
                                     "analysis": example["analysis"],
                                     "queris": example["queris"],
                                     }
                processed_data.append(processed_example)
        else:
            print(f"{mode} is not supported.")
            raise NotImplementedError
    with open(output_file, "w") as fout:
        for line in processed_data:
            fout.write(json.dumps(line) + '\n')


def fact_generation_post_process(input_file, output_file, mode):
    """
    postprocess for fact generation results
    1. parse result;
    """
    data = [json.loads(line.strip()) for line in open(input_file, "r")]
    processed_data = []
    for example in data:
        if mode == "parse_res":
            processed_example = parse_fact_generation_res(example["raw_response"])
            if processed_example is not None:
                example.update(processed_example)
            processed_data.append(example)
        else:
            print(f"{mode} is not supported.")
            raise NotImplementedError
    with open(output_file, "w") as fout:
        for line in processed_data:
            fout.write(json.dumps(line) + '\n')


def test_generation_post_process(input_file, output_file, mode):
    """
    postprocess for test generation results
    1. parse result;
    2. normalize result;
    """
    data = [json.loads(line.strip()) for line in open(input_file, "r")]
    if mode == "parse_res":
        processed_data = []
        for example in data:
            processed_example = parse_test_generation_res(example["raw_response"])
            if processed_example is not None:
                example.update(processed_example)
            processed_data.append(example)
    elif mode == "normalize":
        processed_data = parse_test_generation([x["raw_response"] for x in data])
    else:
        print(f"{mode} is not supported.")
        raise NotImplementedError
    with open(output_file, "w") as fout:
        for line in processed_data:
            fout.write(json.dumps(line) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, help='train/test/test_truth')
    parser.add_argument('--stage', type=str, help='fact_enhance_classify/fact_generation/test_generation')
    args = parser.parse_args()
    print(args)
    global_path = "../data/generation_results"
    if (args.split == "test" or args.split == "test_truth") and args.stage == "fact_enhance_classify":
        # ========== test ========== fact_enhance_classify ==========
        split = args.split
        stage = args.stage
        mode = "parse_res"
        for data_name in ["lima_testset_single_turn_classify",
                          "vicuna_testset_single_turn_classify",
                          "wizardlm_testset_single_turn_classify",
                          "truthfulqa_testset_single_turn_classify"
                          ]:
            input_file = f"{global_path}/{split}/{stage}/{data_name}.jsonl"
            output_file = f"{global_path}/{split}/{stage}/{data_name}_{mode}.jsonl"
            fact_enhance_classify_post_process(input_file, output_file, mode)
        mode = "select_need"
        for data_name in ["lima_testset_single_turn_classify",
                          "vicuna_testset_single_turn_classify",
                          "wizardlm_testset_single_turn_classify",
                          "truthfulqa_testset_single_turn_classify"
                          ]:
            input_file = f"{global_path}/{split}/{stage}/{data_name}.jsonl"
            output_file = f"{global_path}/{split}/{stage}/{data_name}_{mode}.jsonl"
            fact_enhance_classify_post_process(input_file, output_file, mode)
    if (args.split == "test" or args.split == "test_truth") and args.stage == "fact_generation":
        # ========== test ========== fact_generation ==========
        split = args.split
        stage = args.stage
        mode = "parse_res"
        for data_name in ["lima_testset_single_turn_classify_parse_res_select_need_knowledge_gen",
                          "vicuna_testset_single_turn_classify_parse_res_select_need_knowledge_gen",
                          "wizardlm_testset_single_turn_classify_parse_res_select_need_knowledge_gen",
                          "truthfulqa_testset_single_turn_classify_parse_res_select_need_knowledge_gen"
                          ]:
            input_file = f"{global_path}/{split}/{stage}/{data_name}.jsonl"
            output_file = f"{global_path}/{split}/{stage}/{data_name}_{mode}.jsonl"
            fact_generation_post_process(input_file, output_file, mode)
    if (args.split == "test" or args.split == "test_truth") and args.stage == "test_generation":
        # ========== test ========== test_generation ==========
        split = args.split
        stage = args.stage
        mode = "parse_res"
        for data_name in ["lima_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen",
                          "vicuna_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen",
                          "wizardlm_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen",
                          "truthfulqa_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen"
                          ]:
            input_file = f"{global_path}/{split}/{stage}/{data_name}.jsonl"
            output_file = f"{global_path}/{split}/{stage}/{data_name}_{mode}.jsonl"
            test_generation_post_process(input_file, output_file, mode)

        split = args.split
        stage = args.stage
        mode = "normalize"
        for data_name in ["lima_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen",
                          "vicuna_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen",
                          "wizardlm_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen",
                          "truthfulqa_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen"
                          ]:
            input_file = f"{global_path}/{split}/{stage}/{data_name}.jsonl"
            output_file = f"{global_path}/{split}/{stage}/{data_name}_{mode}.jsonl"
            test_generation_post_process(input_file, output_file, mode)
    if args.split == "train" and args.stage == "fact_enhance_classify":
        # ========== train ========== fact_enhance_classify ==========
        split = args.split
        stage = args.stage
        mode = "parse_res"
        for data_name in ["wizardlm_alpaca_single_turn_classify"]:
            input_file = f"{global_path}/{split}/{stage}/{data_name}.jsonl"
            output_file = f"{global_path}/{split}/{stage}/{data_name}_{mode}.jsonl"
            fact_enhance_classify_post_process(input_file, output_file, mode)
        mode = "select_need"
        for data_name in ["wizardlm_alpaca_single_turn_classify_parse_res"]:
            input_file = f"{global_path}/{split}/{stage}/{data_name}.jsonl"
            output_file = f"{global_path}/{split}/{stage}/{data_name}_{mode}.jsonl"
            fact_enhance_classify_post_process(input_file, output_file, mode)
    if args.split == "train" and args.stage == "fact_generation":
        # ========== train ========== fact_generation ==========
        split = args.split
        stage = args.stage
        mode = "parse_res"
        for data_name in ["wizardlm_alpaca_single_turn_classify_parse_res_select_need_knowledge_gen"]:
            input_file = f"{global_path}/{split}/{stage}/{data_name}.jsonl"
            output_file = f"{global_path}/{split}/{stage}/{data_name}_{mode}.jsonl"
            fact_generation_post_process(input_file, output_file, mode)
    if args.split == "train" and args.stage == "test_generation":
        # ========== train ========== test_generation ==========
        split = args.split
        stage = args.stage
        mode = "parse_res"
        for data_name in ["wizardlm_alpaca_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen"]:
            input_file = f"{global_path}/{split}/{stage}/{data_name}.jsonl"
            output_file = f"{global_path}/{split}/{stage}/{data_name}_{mode}.jsonl"
            test_generation_post_process(input_file, output_file, mode)

        split = args.split
        stage = args.stage
        mode = "normalize"
        for data_name in ["wizardlm_alpaca_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen"]:
            input_file = f"{global_path}/{split}/{stage}/{data_name}.jsonl"
            output_file = f"{global_path}/{split}/{stage}/{data_name}_{mode}.jsonl"
            test_generation_post_process(input_file, output_file, mode)

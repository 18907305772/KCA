"""Conduct knowledge inconsistency processing."""

import json


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def construct_data(fact_check_file, test_generation_file, output_file, data_name, model_name, llm_evaluation_result_file, no_fact_type):
    """Construct data for openbook, drop, and sorry."""
    fact_check_data = read_jsonl(fact_check_file)
    test_generation_data = read_jsonl(test_generation_file)
    llm_evaluation_results = json.loads(open(llm_evaluation_result_file, "r").read())
    data_need_facts = []
    for key, value in llm_evaluation_results["threshold_0.6"].items():
        need_fact_idx = int(key.replace(f"idx_{data_name}", ""))
        example = test_generation_data[need_fact_idx]
        if value is True:
            input_value = example["original_input"]["original_input"]["input"]
            answer_value = example["original_input"]["original_input"]["output"]
            output_value = answer_value
            data_need_facts.append({
                "id": f"need_and_{model_name}_have_fact_{need_fact_idx}",
                "conversations": [
                    {
                        "from": "human",
                        "value": input_value,
                    },
                    {
                        "from": "gpt",
                        "value": output_value,
                    }
                ],
                "class": "need_and_have_fact",
                "analysis": example["original_input"]["original_input"]["analysis"],
                "knowledge": example["original_input"]["knowledge"],
            })
        else:
            if no_fact_type == "drop":
                pass
            elif no_fact_type == "openbook":
                input_value = example['original_input']['knowledge'] + "\n" + example["original_input"]["original_input"]["input"]
                answer_value = example["original_input"]["original_input"]["output"]
                output_value = answer_value
                data_need_facts.append({
                    "id": f"need_and_{model_name}_have_no_fact_{need_fact_idx}",
                    "conversations": [
                        {
                            "from": "human",
                            "value": input_value,
                        },
                        {
                            "from": "gpt",
                            "value": output_value,
                        }
                    ],
                    "class": "need_and_have_no_fact",
                    "analysis": example["original_input"]["original_input"]["analysis"],
                    "knowledge": example["original_input"]["knowledge"],
                })
            elif no_fact_type == "sorry":
                input_value = example["original_input"]["original_input"]["input"]
                answer_value = f"Sorry, I don't know the factual information required to answer this question."
                output_value = answer_value
                data_need_facts.append({
                    "id": f"need_and_{model_name}_have_no_fact_{need_fact_idx}",
                    "conversations": [
                        {
                            "from": "human",
                            "value": input_value,
                        },
                        {
                            "from": "gpt",
                            "value": output_value,
                        }
                    ],
                    "class": "need_and_have_no_fact",
                    "analysis": example["original_input"]["original_input"]["analysis"],
                    "knowledge": example["original_input"]["knowledge"],
                })
            else:
                raise NotImplementedError

    data_no_need_facts = []
    no_need_fact_idx = 0
    for example in fact_check_data:
        if "final_prediction" in example and example["final_prediction"] == "<no need>":
            input_value = example["original_input"]["input"]
            answer_value = example["original_input"]["output"]
            output_value = answer_value
            data_no_need_facts.append({
                "id": f"no_need_fact_{no_need_fact_idx}",
                "conversations": [
                    {
                        "from": "human",
                        "value": input_value,
                    },
                    {
                        "from": "gpt",
                        "value": output_value,
                    }
                ],
                "class": "no_need_fact",
                "analysis": example["analysis"],
                "knowledge": "",
            })
            no_need_fact_idx += 1
    final_data = data_need_facts + data_no_need_facts
    print(f"data_need_facts: {len(data_need_facts)}")
    print(f"data_no_need_facts: {len(data_no_need_facts)}")
    print(f"data_final: {len(final_data)}")
    for example in final_data:
        for turn in example["conversations"]:
            assert type(turn["value"]) == str
    with open(output_file, "w") as fout:
        json.dump(final_data, fout, indent=4)


def main():
    global_path = "./data"

    fact_check_file_mapping = {
        "train": ["wizardlm_alpaca_single_turn_classify_parse_res.jsonl"],
        "test": ["lima_testset_single_turn_classify_parse_res.jsonl",
                 "vicuna_testset_single_turn_classify_parse_res.jsonl",
                 "wizardlm_testset_single_turn_classify_parse_res.jsonl"],
        "test_truth": ["truthfulqa_testset_single_turn_classify_parse_res.jsonl"]
    }
    test_generation_file_mapping = {
        "train": ["wizardlm_alpaca_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen_parse_res.jsonl"],
        "test": ["lima_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen_parse_res.jsonl",
                 "vicuna_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen_parse_res.jsonl",
                 "wizardlm_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen_parse_res.jsonl"],
        "test_truth": ["truthfulqa_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen_parse_res.jsonl"]
    }
    llm_evaluation_result_file_mapping = {
        "train": ["wizardlm_alpaca_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen_normalize_sft_instance_behavior.json"],
        "test": ["lima_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen_normalize_sft_instance_behavior.json",
                 "vicuna_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen_normalize_sft_instance_behavior.json",
                 "wizardlm_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen_normalize_sft_instance_behavior.json"],
        "test_truth": ["truthfulqa_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen_normalize_sft_instance_behavior.json"],
    }
    for split in ["train", "test", "test_truth"]:
        for model_name in [ "pythia-6.9b", "llama-2-7b", "mistral-7b-v0.1", "llama-2-13b"]:
            for shot in ["5"]:
                for dataset_idx in range(len(fact_check_file_mapping[split])):
                    for no_fact_type in ["openbook", "drop", "sorry"]:
                        fact_check_file = f"{global_path}/generation_results/{split}/fact_enhance_classify/{fact_check_file_mapping[split][dataset_idx]}"
                        test_generation_file = f"{global_path}/generation_results/{split}/test_generation/{test_generation_file_mapping[split][dataset_idx]}"
                        llm_evaluation_result_file = f"{global_path}/examination/output/{split}/{model_name}/{shot}-shot/{llm_evaluation_result_file_mapping[split][dataset_idx]}"
                        data_name = f"{fact_check_file_mapping[split][dataset_idx].split('_')[0]}_{split}set"
                        output_file = f"{global_path}/processed_results/{model_name}_shot-{shot}_{data_name}_{no_fact_type}.json"
                        try:
                            construct_data(fact_check_file, test_generation_file, output_file, "sft", model_name, llm_evaluation_result_file, no_fact_type)
                        except Exception as e:
                            print(e)


if __name__ == "__main__":
    main()
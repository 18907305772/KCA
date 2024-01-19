"""Obtain examination accuracy."""

import json


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_question_id(idx):
    return idx.replace("_test0", "").replace("_test1", "").replace("_test2", "")


def compute_question_accuracy(data, threshold):
    question_correct = {}
    question_total = {}

    for item in data:
        question_id = extract_question_id(item["idx"])
        correct = item["correct"]

        if question_id not in question_total:
            question_total[question_id] = 0
            question_correct[question_id] = 0

        question_total[question_id] += 1
        if correct == "True":
            question_correct[question_id] += 1

    question_accuracy = {}
    for question_id in question_total:
        accuracy = question_correct[question_id] / question_total[question_id]
        question_accuracy[question_id] = accuracy > threshold
    return question_accuracy


def compute_dataset_accuracy(question_accuracy):
    total_questions = len(question_accuracy)
    correct_questions = sum(1 for q in question_accuracy.values() if q)
    return correct_questions / total_questions


def main():
    dataset_mapping = {
        "train": ["wizardlm_alpaca_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen_normalize"],
        "test": ["lima_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen_normalize",
                 "vicuna_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen_normalize",
                 "wizardlm_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen_normalize",],
        "test_truth": ["truthfulqa_testset_single_turn_classify_parse_res_select_need_knowledge_gen_parse_res_test_gen_normalize"],
    }
    global_path = "./data"
    for split in ["train", "test", "test_truth"]:
        for model_name in ["pythia-6.9b", "llama-2-7b", "mistral-7b-v0.1", "llama-2-13b"]:
            for shot in ["5"]:
                for dataset in dataset_mapping[split]:
                    file_path = f"{global_path}/examination/output/{split}/{model_name}/{shot}-shot/{dataset}.jsonl"
                    sft_instance_behavior = dict()
                    sft_instance_metric = dict()
                    for threshold in [0.3, 0.6, 0.9]:
                        try:
                            data = read_jsonl(file_path)
                            question_accuracy = compute_question_accuracy(data, threshold)
                            dataset_accuracy = compute_dataset_accuracy(question_accuracy)
                            sft_instance_behavior[f'threshold_{threshold}'] = question_accuracy
                            sft_instance_metric[f'threshold_{threshold}'] = dataset_accuracy
                            print(f"sft-instance-level acc (threshold {threshold}): {dataset_accuracy * 100:.2f}%")
                        except Exception as e:
                            print(e)
                    with open(file_path.replace('.jsonl', '_sft_instance_behavior.json'), "w") as fout:
                        json.dump(sft_instance_behavior, fout)
                    with open(file_path.replace('.jsonl', '_sft_instance_metric.json'), "w") as fout:
                        json.dump(sft_instance_metric, fout)


if __name__ == "__main__":
    main()

"""Calculate final evaluation results."""

import json
import os


def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, 'r') as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list


def calculate_effectiveness_statistic(parse_results):
    all_scores = []
    scores = [_["judge_score"] for _ in parse_results]
    classes = [_["class"] for _ in parse_results]
    error_cnt = 0
    result = dict()
    for i in range(len(classes)):
        if scores[i] == -100:
            error_cnt += 1
            continue
        all_scores.append(scores[i])
    result["all_scores"] = sum(all_scores) / len(all_scores) if len(all_scores) != 0 else 0
    result["error_cnt"] = error_cnt
    return result


def calculate_hallucination_classification_statistic(parse_results):
    all_scores = []
    scores = [_["judge_score"] for _ in parse_results]
    classes = [_["class"] for _ in parse_results]
    result = dict()
    error_cnt = 0
    for i in range(len(classes)):
        if scores[i] == -100:
            if parse_results[i]["judge_result"] != "":
                scores[i] = 1
            else:
                error_cnt += 1
                continue
        all_scores.append(scores[i])
        result["all_scores"] = sum(all_scores) / len(all_scores) if len(all_scores) != 0 else 0
        result["error_cnt"] = error_cnt
    return result


def gpt_judge_statistics_func(input_file, judge_type):
    reviews = get_json_list(input_file)
    if judge_type == "effectiveness_judge":
        judge_statistics = calculate_effectiveness_statistic(reviews)
    elif judge_type == "hallucination_judge":
        judge_statistics = calculate_hallucination_classification_statistic(reviews)
    else:
        raise NotImplementedError
    return judge_statistics


if __name__ == "__main__":
    all_statistics = dict()
    for model in ["pythia-6.9b", "llama-2-7b", "mistral-7b-v0.1", "llama-2-13b"]:
        for shot in ["5"]:
            for fact_type in ["baseline", "openbook", "drop", "sorry"]:
                for testset in ["lima_testset", "vicuna_testset", "wizardlm_testset", "truthfulqa_test_truthset"]:
                    for trainset in ["wizardlm_trainset"]:
                        for judge_type in ["hallucination_judge", "effectiveness_judge"]:
                            if fact_type == "baseline":
                                trainset = trainset.replace("wizardlm_trainset", "wizardlm_alpaca_train")
                            global_path = "./evaluation_results/review_greedy"
                            if fact_type == "drop" or fact_type == "openbook" or fact_type == "sorry":
                                input_file = f"{global_path}/data-{model}_shot-{shot}_{testset}_model-{model}_shot-{shot}_{trainset}_{fact_type}_{judge_type}_greedy.jsonl"
                            elif fact_type == "baseline":
                                input_file = f"{global_path}/data-{model}_shot-{shot}_{testset}_model-baseline_{model}_{trainset}_{judge_type}_greedy.jsonl"
                            else:
                                raise NotImplementedError
                            try:
                                judge_statistics = gpt_judge_statistics_func(input_file, judge_type)
                                all_statistics[f"{model}_{shot}_{testset}_{trainset}_{judge_type}_{fact_type}"] = judge_statistics
                            except Exception as e:
                                print(e)
    print(json.dumps(all_statistics, indent=2))
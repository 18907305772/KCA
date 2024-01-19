"""Generate model answer for chat task."""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import ray

from fastchat.model import get_conversation_template


def run_eval(model_path, model_id, conv_temp, question_file, answer_file, num_gpus, do_sample=False):
    if question_file.endswith(".jsonl"):
        ques_jsons = []
        with open(os.path.expanduser(question_file), "r") as ques_file:
            for line in ques_file:
                ques_jsons.append(json.loads(line))
    else:
        ques_jsons = json.loads(open(question_file, "r").read())
    chunk_size = len(ques_jsons) // num_gpus
    ans_handles = []
    for i in range(0, len(ques_jsons), chunk_size):
        ans_handles.append(
            get_model_answers.remote(
                model_path, model_id, conv_temp, ques_jsons[i: i + chunk_size], do_sample
            )
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

    with open(os.path.expanduser(answer_file), "w") as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line) + "\n")


@ray.remote(num_gpus=1)
@torch.inference_mode()
def get_model_answers(model_path, model_id, conv_temp, question_jsons, do_sample=False):
    model_path = os.path.expanduser(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False if "pythia" not in model_path else True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16
    ).cuda()

    ans_jsons = []
    for i, line in enumerate(tqdm(question_jsons)):
        ques_json = line
        idx = ques_json["id"]
        classes = ques_json["class"]
        qs = ques_json["conversations"][0]["value"]
        conv = get_conversation_template(conv_temp)
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt]).input_ids
        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=do_sample,
            temperature=0.7,
            max_new_tokens=1024,
        )
        output_ids = output_ids[0][len(input_ids[0]):]
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        ans_id = shortuuid.uuid()
        ans_jsons.append(
            {
                "id": idx,
                "class": classes,
                "question": qs,
                "answer": outputs,
                "answer_id": ans_id,
                "model_id": model_id,
                "metadata": {"do_sample": do_sample, "temperature": 0.7, "max_new_tokens": 1024},
            }
        )
    return ans_jsons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--conv-temp", type=str, default="vicuna")
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    args = parser.parse_args()

    ray.init()
    run_eval(
        args.model_path,
        args.model_id,
        args.conv_temp,
        args.question_file,
        args.answer_file,
        args.num_gpus,
        args.do_sample,
    )

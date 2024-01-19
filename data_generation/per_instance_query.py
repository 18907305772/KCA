"""Conduct knowledge inconsistency detection."""

import argparse
import json
import time
import os
import asyncio
import openai
from collections import defaultdict
from tqdm.auto import tqdm
from typing import Dict, Tuple, Union, List, Any


DEFAULT_SYSTEM_PROMPT = """
You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2021-09
Current date: 2023-06-01
"""

OPENAI_TEMPERATURE = 1.0


def write_query_log(
        prompt: str,
        res: Dict[str, str],
        out_dir: str,
):
    with open(os.path.join(out_dir, "query_log.jsonl"), "a+", encoding="utf-8") as f_out:
        query_log = {
            "prompt": prompt,
            "res": res,
        }
        json.dump(
            query_log,
            f_out,
            ensure_ascii=False,
        )
        f_out.write("\n")


def query_openai_batch(args):
    extension = args.file_extension if args.file_extension else "jsonl"
    os.makedirs(args.out_dir, exist_ok=True)
    path_in = os.path.join(args.data_dir, args.input)
    path_out = os.path.join(args.out_dir, args.output)
    total_requests, success_num, fail_num, side_info = \
        0, 0, 0, defaultdict(list)
    with open(path_in, encoding="utf-8") as f_in:
        all_lns = f_in.readlines()
        if extension == "jsonl":
            all_content = [json.loads(ln) for ln in all_lns]
        else:
            src_col = "en_task_detail"
            all_content = [{src_col: ln.strip(), } for ln in all_lns]
    instance_num = len(all_content)
    request_batch_size = args.request_batch_size
    total_batch_num = instance_num // request_batch_size + 1
    example_batches = [
        all_content[idx * request_batch_size: (idx + 1) * request_batch_size]
        for idx in range(total_batch_num)
    ]
    progress_bar = tqdm(range(total_batch_num))
    with open(path_out, "w", encoding="utf-8") as f_out:
        for idx in range(total_batch_num):
            batch = example_batches[idx]
            prompt = []
            for exp in batch:
                prompt_tmp = openai_encode_prompt(exp, mode=args.prompt_mode)
                prompt.append(prompt_tmp)
            result: List[Dict[str, str]] = request_openai_batch(prompt)
            if result is not None:
                success_num += len(result)
                for one_prompt, one_res, one_input in zip(prompt, result, batch):
                    write_query_log(prompt=one_prompt, res=one_res, out_dir=args.out_dir)
                    one_res["original_input"] = one_input
                    json.dump(
                        one_res,
                        f_out,
                        ensure_ascii=False
                    )
                    f_out.write("\n")
            else:
                fail_num += 1

            progress_bar.update(1)
    stats_info = {
        "full_planned_requests": total_batch_num,
        "success_num": success_num,
        "fail_num": fail_num,
    }
    return stats_info


def _res_validation(
        res: Dict[str, str],
        mode: str = "turbo"
) -> bool:
    res_text = res["choices"][0]["message"]["content"]
    if type(res_text) is str and len(res_text) > 5:
        return True
    return False


def openai_encode_prompt(
        example: Dict[str, str],
        mode: str = "en_task_to_zh"
) -> str:
    if mode == "fact_enhance_classify_en":
        prompt = open("./prompt_bank/fact_enhance_classify_en.txt").read() + "\n"
        prompt = prompt.format_map(
            {"instruction_to_process": example["input"]}
        )
    elif mode == "fact_generation_en":
        prompt = open("./prompt_bank/fact_generation_en.txt").read() + "\n"
        kw_mapping = {
            "input": example["input"],
            "output": example["output"],
            "analysis": example["analysis"],
            "queries": example["queris"]
        }
        prompt = prompt.format_map(kw_mapping)
    elif mode == "fact_to_tests_en":
        prompt = open("./prompt_bank/fact_to_tests_en.txt").read() + "\n"
        prompt = prompt.format_map(
            {"knowledge": example["knowledge"]}
        )
    else:
        raise NotImplementedError
    return prompt


def request_openai_batch(
        prompt_list: List[str],
):
    max_retry_times, responses = 20, None
    request_sleep_time = 60
    openai_kwargs = {
        "model": "gpt-3.5-turbo-16k-0613",
        "temperature": OPENAI_TEMPERATURE,
        "top_p": 1.0,
        "n": 1,
        "logit_bias": {"50256": -100},
    }
    for trial_idx in range(max_retry_times):
        try:
            messages_list = [[
                {"role": "system",
                 "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user",
                 "content": prompt},
            ] for prompt in prompt_list]
            completions = asyncio.run(
                dispatch_openai_requests(
                    messages_list=messages_list,
                    max_tokens=6000,
                    **openai_kwargs,
                )
            )
            responses = []
            for completion in completions:
                result = completion['choices'][0]
                response = {
                    "raw_response": result.get("message", {}).get("content", ""),
                    "stop_reason": result.get("finish_reason", ),
                }
                responses.append(response)
            return responses
        except Exception as e:
            print(str(e))
            print(f"Trail No. {trial_idx + 1} Failed, now sleep and retrying...")
            time.sleep(request_sleep_time)
    return responses


async def dispatch_openai_requests(
        messages_list: list[list[dict[str, Any]]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        n: int,
        logit_bias: dict,
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.

    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
        n: Return sentence nums.
        logit_bias: logit bias.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            logit_bias=logit_bias,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='input file directory')
    parser.add_argument('--input', '-i', type=str, required=True, help='input file')
    parser.add_argument('--out_dir', type=str, required=True, help='output file directory')
    parser.add_argument('--output', '-o', type=str, required=True, help='output file')
    parser.add_argument('--file_extension', type=str, help='')
    parser.add_argument('--request_batch_size', type=int, default=None, help='prompt batch size for querying openai')
    parser.add_argument('--prompt_mode', type=str, help='')
    args = parser.parse_args()
    print(args)
    query_openai_batch(args)
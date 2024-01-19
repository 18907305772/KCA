"""Use gpt as automatic evaluator for hallucination or effectiveness evaluation."""

import openai
import time
import json
import os
import re
import tqdm
import argparse
import asyncio
from typing import Any, List, Dict, Optional

MAX_API_RETRY = 20


def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, 'r') as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list


async def dispatch_openai_requests(
        messages_list: list[list[dict[str, Any]]],
        model: str,
        temperature: float,
        max_tokens: int,
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.

    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


def get_completion(messages_list: list, model: str, temperature: float = 0.0):
    for i in range(MAX_API_RETRY):
        try:
            completions = asyncio.run(
                dispatch_openai_requests(
                    messages_list=messages_list,
                    model=model,
                    temperature=temperature,
                    max_tokens=1024,
                )
            )
            return completions
        except openai.error.InvalidRequestError:
            print(messages_list)
            print("Error: Invalid Request")
            return None
        except Exception as e:
            print(e)
            time.sleep(30)
    print(f'Failed after {MAX_API_RETRY} retries.')
    raise RuntimeError


def get_prompt_single_score(question, answer, analysis, knowledge, prompt_temp, prompt_type, demos, add_demos):
    prompt = ""
    if add_demos:
        for demo in demos:
            prompt = prompt + demo + "\n"
    if prompt_type == "effectiveness_judge":
        prompt = prompt + prompt_temp.format_map({"question": question, "answer": answer})
    elif prompt_type == "hallucination_judge":
        prompt = prompt + prompt_temp.format_map({"question": question, "answer": answer,
                                                  "analysis": analysis, "knowledge": knowledge})
    return prompt


def post_process_single_score(result: str, prompt_type: str):
    if prompt_type == "hallucination_judge":
        parts = result.strip().split("\n")
        error_cnt = 0
        if len(parts) == 3:
            score = float(parts[0])
            try:
                span = eval(parts[1])
            except:
                span = []
            explain = parts[2]
        else:
            pattern = r"^(\d+)[ \n]+(\[.*\])[ \n]+(.*)$"
            match = re.match(pattern, result)
            if match:
                score = float(match.group(1))
                try:
                    span = eval(match.group(2))
                except:
                    span = []
                explain = match.group(3)
            else:
                score = -100
                span = []
                explain = "none"
                error_cnt += 1
                print(f"Error for parsing, {error_cnt}")
        return score, span, explain
    elif prompt_type == "effectiveness_judge":
        parts = result.strip().split("\n")
        error_cnt = 0
        if len(parts) == 2:
            score = float(parts[0])
            explain = parts[1]
        else:
            pattern = r"^(\d+)[ \n]+(.*)$"
            match = re.match(pattern, result)
            if match:
                score = float(match.group(1))
                explain = match.group(2)
            else:
                score = -100
                explain = "none"
                error_cnt += 1
                print(f"Error for parsing, {error_cnt}")
        if score != -100 and score < 0:
            score = 0.0
        elif score != -100 and score > 10:
            score = 10.0
        return score, None, explain
    else:
        raise NotImplementedError


def get_single_score(input_file, testset_file, output_file, prompt_file, prompt_type=None, use_demo=False,
                     model="gpt-4", temperature=0.0, batch_size=1, no_sorry=False):
    input_examples = get_json_list(input_file)
    testset_examples = json.loads(open(testset_file, "r").read())
    assert len(input_examples) == len(testset_examples)
    for i in range(len(input_examples)):
        input_examples[i]["analysis"] = testset_examples[i]["analysis"]
        input_examples[i]["knowledge"] = testset_examples[i]["knowledge"]
    review_examples = []
    if no_sorry is False:
        for x in input_examples:
            review_examples.append(x)
    else:
        print("Delete sorry answers...")
        for x in input_examples:
            if "Sorry, I don't know the factual information required to answer this question." not in x["answer"]:
                review_examples.append(x)
        print(f"Delete {len(input_examples) - len(review_examples)} examples.")
    if os.path.exists(output_file):
        curr_result = get_json_list(output_file)
    else:
        curr_result = []
    system_prompt = None
    prompt_template = None
    demo_examples = None
    for prompt in get_json_list(prompt_file):
        if prompt["prompt_type"] == prompt_type:
            system_prompt = prompt["system_prompt"]
            prompt_template = prompt["prompt_template"]
            demo_examples = prompt["demo_examples"]
            break
    for i in tqdm.tqdm(range(len(curr_result), len(review_examples), batch_size)):
        examples = review_examples[i: i + batch_size]
        messages_list = []
        for example in examples:
            qs = example["question"]
            ans = example["answer"]
            analysis = example["analysis"]
            knowledge = example["knowledge"]
            prompt = get_prompt_single_score(qs, ans, analysis, knowledge, prompt_template, prompt_type,
                                             demo_examples, use_demo)
            messages_list.append([
                {"role": "system",
                 "content": system_prompt},
                {"role": "user",
                 "content": prompt},
            ])
        completions = get_completion(messages_list, model, temperature)
        if completions:
            try:
                results = [completion['choices'][0]['message']['content'] for completion in completions]
            except:
                print("Error: Not return anything.")
                results = ["" for _ in range(len(messages_list))]
        else:
            results = ["" for _ in range(len(messages_list))]

        parse_results = []
        for result in results:
            score, span, explain = post_process_single_score(result, prompt_type)
            parse_results.append({"judge_result": result,
                                  "judge_score": score,
                                  "judge_span": span,
                                  "judge_explain": explain})
        for idx, example in enumerate(examples):
            example.update(parse_results[idx])
            with open(output_file, "a+") as fout:
                fout.write(json.dumps(example) + '\n')


def check_error_parse(testset_file, output_file, prompt_file, prompt_type=None, use_demo=False,
                      model="gpt-4", temperature=0.0, batch_size=1):
    output_examples = get_json_list(output_file)
    testset_examples = json.loads(open(testset_file, "r").read())
    system_prompt = None
    prompt_template = None
    demo_examples = None
    for prompt in get_json_list(prompt_file):
        if prompt["prompt_type"] == prompt_type:
            system_prompt = prompt["system_prompt"]
            prompt_template = prompt["prompt_template"]
            demo_examples = prompt["demo_examples"]
            break
    for i in tqdm.tqdm(range(0, len(output_examples), batch_size)):
        if output_examples[i]["judge_score"] != -100:
            continue
        examples = output_examples[i: i + batch_size]
        messages_list = []
        for example in examples:
            qs = example["question"]
            ans = example["answer"]
            analysis = example["analysis"]
            knowledge = example["knowledge"]
            prompt = get_prompt_single_score(qs, ans, analysis, knowledge, prompt_template, prompt_type,
                                             demo_examples, use_demo)
            messages_list.append([
                {"role": "system",
                 "content": system_prompt},
                {"role": "user",
                 "content": prompt},
            ])
        completions = get_completion(messages_list, model, temperature)
        if completions:
            try:
                results = [completion['choices'][0]['message']['content'] for completion in completions]
            except:
                print("Error: Not return anything.")
                results = ["" for _ in range(len(messages_list))]
        else:
            results = ["" for _ in range(len(messages_list))]

        parse_results = []
        for result in results:
            score, span, explain = post_process_single_score(result, prompt_type)
            parse_results.append({"judge_result": result,
                                  "judge_score": score,
                                  "judge_span": span,
                                  "judge_explain": explain})
        for idx, example in enumerate(examples):
            example.update(parse_results[idx])

    with open(output_file, "w") as fout:
        for example in output_examples:
            fout.write(json.dumps(example) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_file", type=str, default="answer.jsonl")
    parser.add_argument("--testset_file", type=str, default="testset.jsonl")
    parser.add_argument("--review_file", type=str, default="review.jsonl")
    parser.add_argument("--prompt_file", type=str, default="prompt.jsonl")
    parser.add_argument("--prompt_type", type=str, default="none")
    parser.add_argument("--use_demo", action="store_true")
    parser.add_argument("--review_model", type=str, default="gpt-4")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--check_error_parse", action="store_true")
    parser.add_argument("--no_sorry", action="store_true")
    args = parser.parse_args()
    print(args)
    if not args.check_error_parse:
        get_single_score(args.answer_file,
                         args.testset_file,
                         args.review_file,
                         args.prompt_file,
                         prompt_type=args.prompt_type,
                         use_demo=args.use_demo,
                         model=args.review_model,
                         temperature=0.0,
                         batch_size=args.batch_size,
                         no_sorry=args.no_sorry)
    else:
        check_error_parse(args.testset_file,
                          args.review_file,
                          args.prompt_file,
                          prompt_type=args.prompt_type,
                          use_demo=args.use_demo,
                          model=args.review_model,
                          temperature=0.0,
                          batch_size=1)


if __name__ == "__main__":
    main()
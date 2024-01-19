import argparse
import os
import torch
import random
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Dict
from tqdm import tqdm
from examination.utils import get_next_word_predictions, load_hf_lm_and_tokenizer, query_openai_chat_model, dynamic_import_function

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(
        examples: List[Dict[str, str]],
        idx: int,
        include_answer: bool = True
) -> str:
    example = examples[idx]
    prompt = example["question"]
    k = len(example["options"])
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], example["options"][j])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(example["answer"])
    return prompt


def gen_prompt(
        examples: List[Dict[str, str]],
        subject: str,
        k: int = -1,

) -> str:
    prompt = "The following are multiple choice questions" \
             " (with answers) about factual knowledge.\n\n"
    if k == -1:
        k = len(examples)
    for i in range(k):
        prompt += format_example(examples, i)
    return prompt


@torch.no_grad()
def eval_hf_model(
        args: argparse.ArgumentParser,
        subject: str,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        dev_set: List[Dict[str, str]],
        test_set: List[Dict[str, str]],
        batch_size: int = 1
) -> Tuple[np.ndarray, float, np.ndarray]:
    prompts = []
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    for i in range(0, len(test_set)):
        k = args.ntrain
        prompt_end = format_example(test_set, i, include_answer=False)
        train_prompt = gen_prompt(dev_set, subject, k)
        prompt = train_prompt + prompt_end

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "The answer is:"
            else:
                prompt += " The answer is:"

        tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        while len(tokenized_prompt) > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_set, subject, k)
            prompt = train_prompt + prompt_end

            if args.use_chat_format:
                messages = [{"role": "user", "content": prompt}]
                prompt = chat_formatting_function(messages, add_bos=False)
                if prompt[-1] in ["\n", " "]:
                    prompt += "The answer is:"
                else:
                    prompt += " The answer is:"

            tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        prompts.append(prompt)

    answer_choice_ids = [tokenizer.encode(" " + answer_choice, add_special_tokens=False)[-1] for answer_choice in choices]
    pred_indices, all_probs = get_next_word_predictions(
        model, tokenizer, prompts, candidate_token_ids=answer_choice_ids, return_token_predictions=False,
        batch_size=batch_size
    )

    cors = []
    ground_truths = [exp["answer"] for exp in test_set]
    for i in range(len(pred_indices)):
        prediction = choices[pred_indices[i]]
        ground_truth = ground_truths[i]
        cors.append(prediction == ground_truth)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs


def eval_openai_chat_engine(
        args,
        subject,
        engine,
        dev_set,
        test_set,
        batch_size=1
) -> Tuple[np.ndarray, float, np.ndarray]:
    import tiktoken
    gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
    answer_choice_ids = [gpt_tokenizer.encode(" " + x)[0] for x in choices]
    prompts = []
    for i in range(0, len(test_set)):
        k = args.ntrain
        prompt_end = format_example(test_set, i, include_answer=False)
        train_prompt = gen_prompt(dev_set, subject, k)
        prompt = train_prompt + prompt_end
        prompts.append(prompt)

    instances = [{"id": prompt, "prompt": prompt} for _, prompt in enumerate(prompts)]
    results = query_openai_chat_model(
        engine=engine,
        instances=instances,
        batch_size=batch_size if batch_size else 10,
        output_path=os.path.join(args.save_dir, f"{subject}_openai_results.jsonl"),
        logit_bias={token_id: 100 for token_id in answer_choice_ids},
        max_tokens=1,
        retry_limit=20,
    )
    cors = []
    ground_truths = [exp["answer"] for exp in test_set]
    for i in range(len(test_set)):
        prediction = results[i]["output"].strip()
        ground_truth = ground_truths[i]
        cors.append(prediction == ground_truth)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array([[0.25, 0.25, 0.25, 0.25] for _ in range(len(test_set))])

    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs


def main(args):
    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=args.gptq,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
    subjects = sorted(
        [
            f.split("_test.jsonl")[0]
            for f in os.listdir(args.data_dir)
            if "_test.jsonl" in f
        ]
    )
    print("evaluate on the subjects: ", subjects)

    if args.subjects:
        assert all(subj in subjects for subj in
                   args.subjects), f"Some of the subjects you specified are not valid: {args.subjects}"
        subjects = args.subjects

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    all_cors, subject_cors = [], dict()
    for subject in tqdm(subjects, desc="Evaluating subjects: "):
        dev_set = [
            {"question": "Which aspect of system security is evaluated in a physical security assessment?",
             "options": ["Email security controls", "Patch management processes", "Network security controls",
                         "Physical security measures"],
             "answer": "D"},
            {"question": "Which Python libraries are commonly used for sleep data analysis?",
             "options": ["Pandas, Matplotlib, and Seaborn", "NumPy, Scikit-learn, and Plotly",
                         "TensorFlow, Keras, and PyTorch", "Django, Flask, and SQLAlchemy"],
             "answer": "A"},
            {"question": "What does sentiment polarity measure in sentiment analysis?",
             "options": ["The degree of personal opinion expressed in a text.",
                         "The strength and direction of sentiment in a text.", "The subjectivity score of a statement.",
                         "The emotional connotations associated with specific words."],
             "answer": "B"},
            {
                "question": "Which of the following is the recommended operating system for a Raspberry Pi when setting up a personal cloud server?",
                "options": ["Ubuntu", "Raspbian", "Fedora", "Arch Linux"],
                "answer": "B"},
            {"question": "One of the benefits of vegetation in woodlands streams for fish populations is:",
             "options": ["Providing hiding places and cover from predators.",
                         "Absorbing excess nutrients and pollutants.", "Regulating water temperature.",
                         "Enhancing the growth of algae."],
             "answer": "A"}
        ]
        test_set = [json.loads(ln) for ln in open(os.path.join(args.data_dir, subject + "_test.jsonl")).readlines()]
        if args.n_instances and args.n_instances < len(test_set):
            test_set = random.sample(test_set, args.n_instances)
        if args.model_name_or_path:
            cors, acc, probs = eval_hf_model(
                args,
                subject,
                model,
                tokenizer,
                dev_set,
                test_set,
                args.eval_batch_size
            )
        else:
            cors, acc, probs = eval_openai_chat_engine(
                args,
                subject,
                args.openai_engine,
                dev_set,
                test_set,
                args.eval_batch_size
            )
        all_cors.append(cors)
        subject_cors[subject] = np.mean(cors)
        for test_sample, cor, prob in zip(test_set, list(cors), list(probs)):
            test_sample["correct"] = str(cor)
            for j in range(len(prob)):
                choice = choices[j]
                test_sample["choice{}_probs".format(choice)] = prob[j]

        with open(os.path.join(args.save_dir, "{}.jsonl".format(subject)), "w") as f_out:
            for exp in test_set:
                f_out.write(json.dumps(exp, ensure_ascii=False))
                f_out.write("\n")

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "average_acc": weighted_acc,
                "subject_acc": subject_cors,
            },
            f,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ntrain",
        type=int,
        default=5
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/mmlu"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/mmlu/llama-7B/"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="if specified, we will use the OpenAI API to generate the predictions."
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        help="which subjects to evaluate. If not specified, all the 57 subjects will be evaluated."
    )
    parser.add_argument(
        "--n_instances",
        type=int,
        help="if specified, a maximum of n_instances per subject will be used for the evaluation."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    args = parser.parse_args()

    assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)

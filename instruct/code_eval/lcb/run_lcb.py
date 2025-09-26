import os
import json
import types

from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.utils.path_utils import (
    ensure_dir,
    get_cache_path,
)
from lcb_runner.utils.extraction_utils import extract_code
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.evaluation.compute_code_generation_metrics import evaluate_generations

from lcb_runner.lm_styles import LanguageModel, LMStyle

from lcb_runner.runner.scenario_router import (
    combine_results,
    sort_and_extract_save_results,
    get_metrics,
)
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem, Difficulty
from lcb_runner.prompts.code_generation import (
    get_base_model_question_template_answer,
    PromptConstants,
)
from transformers import AutoModel, AutoTokenizer
from diffusion_utils.dream_generation import DreamGenerationWrapper
from diffusion_utils.llada_generation import generate as llada_generate
import torch
import argparse
from functools import partial
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-0301",
        help="Name of the model to use matching `lm_styles.py`",
    )
    parser.add_argument(
        "--scenario",
        type=Scenario,
        default=Scenario.codegeneration,
        help="Type of scenario to run",
    )
    parser.add_argument(
        "--release_version",
        type=str,
        default="release_latest",
        help="whether to use full set of tests (slower and more memory intensive evaluation)",
    )
    parser.add_argument(
        "--cot_code_execution",
        action="store_true",
        help="whether to use CoT in code execution scenario",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument(
        "--stop",
        default="",
        type=str,
        help="Stop token (use `,` to separate multiple tokens)",
    )
    parser.add_argument("--continue_existing", action="store_true")
    parser.add_argument("--continue_existing_with_eval", action="store_true")
    parser.add_argument(
        "--use_cache", action="store_true", help="Use cache for generation"
    )
    parser.add_argument(
        "--add_prefix", action="store_true", help="Add prefix to the prompt"
    )
    parser.add_argument(
        "--fast_dllm", action="store_true", help="Use fast_dllm for generation"
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the results")
    parser.add_argument(
        "--num_process_evaluate",
        type=int,
        default=12,
        help="Number of processes to use for evaluation",
    )
    parser.add_argument("--timeout", type=int, default=6, help="Timeout for evaluation")
    parser.add_argument(
        "--custom_output_file",
        type=str,
        default=None,
        help="Path to the custom output file used in `custom_evaluator.py`",
    )
    parser.add_argument(
        "--custom_output_save_name",
        type=str,
        default=None,
        help="Folder name to save the custom output results (output file folder modified if None)",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Dtype for vllm")
    # Added to avoid running extra generations (it's slow for reasoning models)
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start date for the contest to filter the evaluation file (format - YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date for the contest to filter the evaluation file (format - YYYY-MM-DD)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        help="Difficulty for the contest to filter the evaluation file (format - easy, medium, hard)",
    )
    parser.add_argument(
        "--use_instruct_prompt",
        action="store_true",
        help="Use instruct prompt for diffusion",
        default=False,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for diffusion",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for sampling"
    )
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p for sampling")
    parser.add_argument(
        "--max_tokens", type=int, default=2048, help="Max tokens for sampling"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max generation length for sampling",
    )
    parser.add_argument(
        "--diffusion_steps",
        type=int,
        default=512,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--diffusion_remask_alg",
        type=str,
        default="origin",
        help="Algorithm for diffusion remasking",
    )
    parser.add_argument(
        "--diffusion_remask_alg_temp",
        type=float,
        default=None,
        help="Temperature for diffusion remasking",
    )

    args = parser.parse_args()
    if args.stop:
        args.stop = args.stop.split(",")
    else:
        args.stop = []

    return args


def prompt_formatter(
    question: CodeGenerationProblem,
    tokenizer=None,
    use_instruct_prompt: bool = False,
    add_prefix: bool = False,
):
    if use_instruct_prompt:
        prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
        prompt += f"Question:\n{question.question_content}\n\n"
        if question.starter_code:
            prompt += f"{PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
            prompt += f"```python\n{question.starter_code}\n```\n\n"
        else:
            prompt += f"{PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n\n"
            prompt += f"```python\n# YOUR CODE HERE\n```\n\n"

        messages = [
            {"role": "user", "content": prompt},
        ]

        if add_prefix:
            _MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"
            prefix = f"```python\n{_MAGIC_SPLITTER_}\n```"

            messages += [
                {"role": "assistant", "content": prefix},
            ]

        if tokenizer.chat_template is None:
            prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{prompt.strip()}
<|im_end|>
<|im_start|>assistant
{prefix if add_prefix else ''}"""
        else:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=not add_prefix,
                truncation=False,
                padding=False,
            )

        if add_prefix:
            prompt = prompt.split(_MAGIC_SPLITTER_)[0]

        return prompt
    else:
        return get_base_model_question_template_answer(question)


def get_reward(results, benchmark, num_process_evaluate=16, timeout=6):
    def get_public_test_cases(instance):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [t.input for t in instance.public_test_cases],
                    "outputs": [t.output for t in instance.public_test_cases],
                    "fn_name": instance.metadata.get("func_name", None),
                }
            ),
        }

    public_test_cases = [get_public_test_cases(instance) for instance in benchmark]
    generations = [
        [output.strip() for output in outputs_list] for outputs_list in results
    ]

    # Get evaluation results using the existing codegen_metrics function
    results, metadata = evaluate_generations(
        public_test_cases,
        generations,
        debug=False,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )
    instance_grades = extract_instance_results(results)
    # Convert to reward format: 1 for pass, -1 for fail
    rewards = [1 if all(instance_grade) else -1 for instance_grade in instance_grades]
    return rewards


class DiffusionRunner:
    def __init__(self, args, model: LanguageModel, cache_generation_path: str):
        self.args = args
        self.model_dict = model
        model_tokenizer_path = model.model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = AutoModel.from_pretrained(
            model_tokenizer_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_tokenizer_path,
            trust_remote_code=True,
        )
        print("Diffusion sampling parameters:")
        print(f"max_new_tokens: {args.max_new_tokens}")
        print(f"steps:          {args.diffusion_steps}")
        print(f"temperature:    {args.temperature}")
        print(f"top_p:          {args.top_p}")
        print(f"alg:            {args.diffusion_remask_alg}")
        print(f"alg_temp:       {args.diffusion_remask_alg_temp}")
        print(f"fast_dllm:      {args.fast_dllm}")
        if model_tokenizer_path.startswith("GSAI-ML/LLaDA-8B"):
            self._generate_fn = partial(
                llada_generate,
                model=self._model,
                steps=args.diffusion_steps,
                gen_length=args.max_new_tokens,
                temperature=args.temperature,
            )
        elif model_tokenizer_path.startswith("apple/DiffuCoder"):
            self._model = AutoModel.from_pretrained(
                model_tokenizer_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            self.diffusion_generation_wrapper = self._model
            self._generate_fn = partial(
                self.diffusion_generation_wrapper.diffusion_generate,
                max_new_tokens=args.max_new_tokens,
                output_history=False,
                return_dict_in_generate=False,
                steps=args.diffusion_steps,
                temperature=args.temperature,
                top_p=args.top_p,
                alg=args.diffusion_remask_alg,
                alg_temp=args.diffusion_remask_alg_temp,
            )
        else:
            # NOTE: this is for our upcoming model xD
            if args.fast_dllm:
                from diffusion_utils.fast_dllm.modeling_dream import DreamModel

                self._model = DreamModel.from_pretrained(
                    model_tokenizer_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                ).to(self.device)
                from diffusion_utils.fast_dllm.generation_utils_block import (
                    DreamGenerationMixin,
                )

                self._model.diffusion_generate = types.MethodType(
                    DreamGenerationMixin.diffusion_generate, self._model
                )
                self._model._sample = types.MethodType(
                    DreamGenerationMixin._sample, self._model
                )
            else:
                self._model = AutoModel.from_pretrained(
                    model_tokenizer_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                ).to(self.device)
            # self.diffusion_generation_wrapper = DreamGenerationWrapper(self._model)
            self.diffusion_generation_wrapper = self._model
            self._generate_fn = partial(
                self.diffusion_generation_wrapper.diffusion_generate,
                max_new_tokens=args.max_new_tokens,
                output_history=False,
                return_dict_in_generate=False,
                steps=args.diffusion_steps,
                temperature=args.temperature,
                top_p=args.top_p,
                alg=args.diffusion_remask_alg,
                alg_temp=args.diffusion_remask_alg_temp,
            )

        if self.args.use_cache:
            self.cache_path = cache_generation_path
            if os.path.exists(self.cache_path):
                with open(self.cache_path) as f:
                    self.cache: dict = json.load(f)
            else:
                self.cache = {}
        else:
            self.cache_path = None
            self.cache = None

    def save_cache(self):
        if self.args.use_cache:
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f, indent=4)

    def run_main(self, benchmark: list, format_prompt: callable) -> list[list[str]]:
        outputs = []

        for problem in tqdm(benchmark):
            prompt = format_prompt(problem)

            print("########################")
            print(prompt, flush=True)
            print("########################")

            if isinstance(prompt, list):
                prompt_cache = json.dumps(prompt)
            elif isinstance(prompt, tuple):
                prompt_cache = prompt[0] + json.dumps(prompt[1])
            else:
                prompt_cache = prompt

            output = None
            if self.cache is not None and prompt_cache in self.cache:
                if len(self.cache[prompt_cache]) == self.args.n:
                    output = self.cache[prompt_cache]

            if output is None:
                output = self._diffusion_generate(prompt, problem)
                assert len(output) == self.args.n
            print("--------------------------")
            print(output, flush=True)
            print("--------------------------")
            outputs.append(output)

            if self.args.use_cache:
                self.cache[prompt_cache] = output  ## save the output to cache
                self.save_cache()
        return outputs

    def _diffusion_generate(
        self, prompt: str, problem: CodeGenerationProblem
    ) -> list[str]:
        inputs = self._tokenizer(
            prompt, padding=True, truncation=False, return_tensors="pt"
        ).input_ids.to(device=self.args.device)

        def reward_fn(outputs):
            gen_strs = self._tokenizer.batch_decode(
                outputs[:, inputs.size(-1) :],
            )
            gen_strs = [extract_code(g, self.model_dict.model_style) for g in gen_strs]
            rewards = get_reward(
                [gen_strs],
                [problem],
                num_process_evaluate=1,
            )
            return rewards

        generations = self._generate_fn(inputs, reward_fn=reward_fn)
        generations = self._tokenizer.batch_decode(generations[:, inputs.shape[1] :])
        for i, g in enumerate(generations):
            for s in self.args.stop:
                g = g.split(s)[0]
            generations[i] = g

        if self.args.add_prefix:
            generations = ["```python\n" + g for g in generations]

        return generations


def diffusion_get_output_path(model_repr: str, args) -> str:
    n = args.n
    temperature = args.temperature
    diffusion_steps = args.diffusion_steps
    max_new_tokens = args.max_new_tokens
    diffusion_remask_alg = args.diffusion_remask_alg
    path = f"output_nostop_fixdecode{'_addprefix' if args.add_prefix else ''}/{model_repr}/sample{n}_{temperature}_{diffusion_steps}_{max_new_tokens}_{diffusion_remask_alg}.json"
    if args.fast_dllm:
        path = path.replace("/sample", "/sample_fast_dllm")
    ensure_dir(path)
    return path


def load_code_generation_dataset_with_difficulty(
    release_version="release_v1", start_date=None, end_date=None, difficulty=None
) -> list[CodeGenerationProblem]:
    dataset = load_dataset(
        "livecodebench/code_generation_lite",
        split="test",
        version_tag=release_version,
        trust_remote_code=True,
    )
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]

    if difficulty is not None:
        dataset = [e for e in dataset if e.difficulty == Difficulty(difficulty)]

    print(f"Loaded {len(dataset)} problems")
    return dataset


def main():
    args = get_args()
    assert args.n == 1, "Only decode 1 sample at a time for now"
    model = LanguageModel(
        model_name=args.model,
        model_repr=args.model.replace("/", "_"),
        model_style=(
            LMStyle.DreamInstruct if args.use_instruct_prompt else LMStyle.GenericBase
        ),
        release_date=None,
        link=None,
    )
    benchmark = load_code_generation_dataset_with_difficulty(
        args.release_version, args.start_date, args.end_date, args.difficulty
    )
    if args.debug:
        print(f"Running with {len(benchmark)} instances in debug mode")
        benchmark = benchmark[:15]

    output_path = diffusion_get_output_path(model.model_repr, args)
    cache_generation_path = output_path.replace(".json", "_cache.json")
    eval_file = output_path.replace(".json", "_eval.json")
    eval_all_file = output_path.replace(".json", "_eval_all.json")

    if args.continue_existing or args.continue_existing_with_eval:
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                old_save_results = json.load(f)
        elif os.path.exists(eval_all_file):
            with open(eval_all_file, "r") as f:
                old_save_results = json.load(f)
        else:
            print(
                f"File {output_path} does not exist in --continue_existing, starting from scratch"
            )
            old_save_results = []

        old_save_results = [
            instance
            for instance in old_save_results
            if instance["output_list"]
            # and [x for x in instance["output_list"] if x]
        ]
        old_save_results_question_ids = [
            instance["question_id"] for instance in old_save_results
        ]
        remaining_benchmark = [
            instance
            for instance in benchmark
            if instance.question_id not in old_save_results_question_ids
        ]
        print(
            f"Found {len(old_save_results)} existing generations, continuing with {len(remaining_benchmark)} remaining"
        )
    else:
        old_save_results = []
        remaining_benchmark = benchmark

    if len(remaining_benchmark) > 0:
        runner = DiffusionRunner(args, model, cache_generation_path)
        prompt_format_fn = partial(
            prompt_formatter,
            tokenizer=runner._tokenizer,
            use_instruct_prompt=args.use_instruct_prompt,
            add_prefix=args.add_prefix,
        )
        results: list[list[str]] = runner.run_main(
            remaining_benchmark,
            prompt_format_fn,
        )
    else:
        results = []

    combined_results = combine_results(
        args.scenario, results, model, args.cot_code_execution
    )

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(
            remaining_benchmark, combined_results
        )
    ]

    if args.continue_existing or args.continue_existing_with_eval:
        save_results += old_save_results

    save_results, _ = sort_and_extract_save_results(args.scenario, save_results)

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)

    if args.evaluate:
        if args.continue_existing_with_eval and os.path.exists(eval_all_file):
            with open(eval_all_file) as fp:
                old_eval_all_results = json.load(fp)

            if os.path.exists(eval_file):
                with open(eval_file) as fp:
                    old_eval_results = json.load(fp)
            else:
                old_eval_results = None

            old_eval_results_question_ids = [
                instance["question_id"] for instance in old_eval_all_results
            ]
            remaining_indices = [
                idx
                for idx in range(len(benchmark))
                if benchmark[idx].question_id not in old_eval_results_question_ids
            ]
            benchmark = [benchmark[idx] for idx in remaining_indices]
            combined_results = [combined_results[idx] for idx in remaining_indices]

            old_eval_size = len(old_eval_results_question_ids)
            new_eval_size = len(benchmark)

            if new_eval_size == 0:
                return

            print(f"Found {old_eval_size}, running evals for {new_eval_size} problems")

            metrics = get_metrics(args.scenario, args, benchmark, combined_results)
            graded = extract_instance_results(metrics[1])

            if old_eval_results:
                for key in metrics[0]:
                    if key in old_eval_results[0]:
                        if key != "detail":
                            metrics[0][key] = (
                                old_eval_size * old_eval_results[0][key]
                                + new_eval_size * metrics[0][key]
                            )
                            metrics[0][key] /= old_eval_size + new_eval_size

                for key in metrics[0]["detail"]:
                    if key in old_eval_results[0]["detail"]:
                        metrics[0]["detail"][key] = {
                            **metrics[0]["detail"][key],
                            **old_eval_results[0]["detail"][key],
                        }
                metrics[1] = {**metrics[1], **old_eval_results[1]}
            else:
                print("Old eval file not present, cannot update eval file")
                metrics = {}

        else:
            metrics = get_metrics(args.scenario, args, benchmark, combined_results)
            graded = extract_instance_results(metrics[1])
            old_eval_all_results = []
            old_eval_results = []

        if args.scenario == Scenario.codegeneration:
            if metrics:
                metadatas = metrics[2]
            else:
                metadatas = [[] for _ in benchmark]
            save_eval_results = [
                instance.insert_output_evaluation(
                    outputs_list, extracted_list, graded_list, metadata=meta
                )
                for instance, (outputs_list, extracted_list), graded_list, meta in zip(
                    benchmark, combined_results, graded, metadatas
                )
            ]
            if metrics and old_eval_results:
                old_eval_results
                metrics[2] = old_eval_results[2] + metrics[2]
        else:
            raise ValueError(f"Scenario {args.scenario} not supported")

        save_eval_results = old_eval_all_results + save_eval_results

        with open(eval_file, "w") as f:
            json.dump(metrics, f, indent=4)

        with open(eval_all_file, "w") as f:
            json.dump(save_eval_results, f, indent=4)


if __name__ == "__main__":
    main()

# Copyright (2025) critic-rl Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from src.eval.code_utils import desanitize
from src.eval.codeio_utils import compute_score as codeio_score
from src.eval.sandbox_utils import get_submit_fn, submit_to_sandbox


async def verify_code_contests_sample(model_output, info):
    _, is_valid = desanitize(model_output)
    if not is_valid:
        return False, {}
    req = get_submit_fn("code_contests")(model_output, info)
    res = await submit_to_sandbox(req)
    return res.accepted, res.dict() | {
        "is_valid": is_valid,
        "model_output": model_output,
    }


async def verify_livecodebench_sample(model_output, info):
    req = get_submit_fn("livecodebench")(model_output, info)
    res = await submit_to_sandbox(req)
    return res.accepted, res.dict()


async def verify_mbppplus_sample(model_output, info):
    _, is_valid = desanitize(model_output)
    if not is_valid:
        return False, {}
    req = get_submit_fn("mbppplus")(model_output, info)
    res = await submit_to_sandbox(req)
    return res.status == "Success", res.dict() | {
        "is_valid": is_valid,
        "model_output": model_output,
    }


async def verify_autoeval_sample(model_output, info):
    _, is_valid = desanitize(model_output)
    if not is_valid:
        return False, {}
    req = get_submit_fn("autoeval")(model_output, info)
    res = await submit_to_sandbox(req)
    return res.status == "Success", res.dict() | {
        "is_valid": is_valid,
        "model_output": model_output,
    }


async def apply_verifiable_reward(response: str, dataset: str, info: dict):
    if dataset.lower() == "code_contests":
        reward, metadata = await verify_code_contests_sample(response, info)
        metadata["dataset"] = "code_contests"
    elif dataset.lower() == "livecodebench":
        reward, metadata = await verify_livecodebench_sample(response, info)
        metadata["dataset"] = "livecodebench"
    elif dataset.lower() == "mbppplus":
        reward, metadata = await verify_mbppplus_sample(response, info)
        metadata["dataset"] = "mbppplus"
    elif dataset.lower() == "autoeval":
        reward, metadata = await verify_autoeval_sample(response, info)
        metadata["dataset"] = "autoeval"
    elif dataset.lower() == "codeio":
        reward, metadata = codeio_score(response, info["ground_truth"])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return reward, metadata

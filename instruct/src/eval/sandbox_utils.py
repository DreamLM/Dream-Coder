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
import asyncio
import logging
import os
import random
import re
from typing import Optional, Union

import numpy as np
from sandbox_fusion import (
    EvalResult,
    RunCodeRequest,
    RunCodeResponse,
    RunStatus,
    SubmitRequest,
    TestConfig,
    run_code_async,
    submit_async,
)

from src.eval.code_utils import desanitize

logger = logging.getLogger(__name__)
TIMEOUT = 320
RUN_TIMEOUT = 10
SANDBOX_FUSION_ENDPOINT = os.getenv("SANDBOX_FUSION_ENDPOINT", "http://localhost:8080")


def parse_endpoints(endpoint_spec):
    # Handle simple case without brackets
    if "[" not in endpoint_spec:
        return [endpoint_spec]

    # Extract base name and bracket content
    match = re.match(r"([^[]+)\[([^\]]+)\]", endpoint_spec)
    if not match:
        return [endpoint_spec]

    base_name = match.group(1)
    bracket_content = match.group(2)

    endpoints = []

    # Split by comma to handle multiple ranges/individual numbers
    parts = bracket_content.split(",")

    for part in parts:
        part = part.strip()

        # Check if it's a range (contains hyphen)
        if "-" in part:
            # Handle range like "15-18"
            range_match = re.match(r"(\d+)-(\d+)", part)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))

                # Determine padding based on the original format
                start_str = range_match.group(1)
                end_str = range_match.group(2)
                padding = max(len(start_str), len(end_str))

                for i in range(start, end + 1):
                    endpoints.append(f"{base_name}{str(i).zfill(padding)}")
        else:
            # Handle individual number
            # Preserve original padding if it exists
            if part.isdigit():
                padding = len(part)
                endpoints.append(f"{base_name}{part.zfill(padding)}")

    return endpoints


def get_submit_fn(dataset_name: str):
    def get_run_timeout(info):
        run_timeout = info.get("time_limit", RUN_TIMEOUT)
        if np.isnan(run_timeout):
            run_timeout = RUN_TIMEOUT
        return run_timeout

    if dataset_name == "code_contests":

        def submit_fn(response, info):
            provided_data = {
                "test": info["test"],
            }

            req = SubmitRequest(
                dataset="code_contests",
                id=0,
                config=TestConfig(
                    language="python",
                    dataset_type="CommonOJDataset",
                    provided_data=provided_data,
                    run_timeout=get_run_timeout(info),
                ),
                completion=response,
            )

            return req

    elif dataset_name == "livecodebench":

        def submit_fn(response, info):
            provided_data = {k: info[k] for k in ["id", "content", "labels", "test"]}
            provided_data
            req = SubmitRequest(
                dataset="dataset_id",
                id=0,
                config=TestConfig(
                    dataset_type="LiveCodeBenchDataset",
                    provided_data=provided_data,
                    run_timeout=get_run_timeout(info),
                ),
                completion=response,
            )

            return req

    elif dataset_name == "mbppplus":

        def submit_fn(response, info):
            if isinstance(info["test"], list):
                ut = "\n".join(info["test"])
            else:
                ut = info["test"]
            req = RunCodeRequest(
                **{
                    "code": desanitize(response)[0].strip() + "\n" + ut,
                    "language": "python",
                    "run_timeout": get_run_timeout(info),
                }
            )

            return req

    elif dataset_name == "autoeval":

        def submit_fn(response, info):
            req = RunCodeRequest(
                **{
                    "code": desanitize(response)[0].strip() + "\n\n" + info["test"],
                    "language": "pytest",
                    "run_timeout": RUN_TIMEOUT,
                }
            )
            return req

    else:
        raise NotImplementedError(f"dataset {dataset_name} not supported")
    return submit_fn


async def submit_to_sandbox(
    request: Union[RunCodeRequest, SubmitRequest],
) -> Union[RunCodeResponse, EvalResult]:
    candidate_endpoints = parse_endpoints(SANDBOX_FUSION_ENDPOINT)
    endpoint = random.choice(candidate_endpoints)

    fn = run_code_async if isinstance(request, RunCodeRequest) else submit_async
    try:
        resp = await fn(request, endpoint=endpoint, client_timeout=TIMEOUT)
        return resp
    except (asyncio.TimeoutError, TimeoutError, Exception) as e:
        logger.warning(f"Request to sandbox timed out or failed: {e}")
        print(f"Request to sandbox timed out or failed: {request}")

        # Return different default failure responses based on request type
        if isinstance(request, RunCodeRequest):
            return RunCodeResponse(
                status=RunStatus.SandboxError,
                message="Request timed out",
            )
        else:
            return EvalResult(
                id=request.id, accepted=False, extracted_code="", tests=[]
            )

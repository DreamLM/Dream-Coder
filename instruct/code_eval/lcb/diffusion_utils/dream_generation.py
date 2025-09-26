# coding=utf-8
import warnings
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

logger = logging.get_logger(__name__)


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    return confidence, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", "origin")
        self.alg_temp: Optional[float] = kwargs.pop(
            "alg_temp", None
        )  # for maskgit_plus

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop(
            "return_dict_in_generate", False
        )
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass


class DreamGenerationWrapper:
    def __init__(self, model):
        self.model = model
        self.device = model.device

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = (
                generation_config.max_new_tokens + input_ids_length
            )

        elif has_default_max_length:  # by default let's always generate 20 new tokens
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = (
                    generation_config.max_length + input_ids_length
                )
                max_position_embeddings = getattr(
                    self.model.config, "max_position_embeddings", None
                )
                if max_position_embeddings is not None:
                    generation_config.max_length = min(
                        generation_config.max_length, max_position_embeddings
                    )

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(
                self.model.config
            )
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = (
                        self.model.generation_config.bos_token_id
                    )
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = (
                        self.model.generation_config.eos_token_id
                    )
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = (
                        self.model.generation_config.pad_token_id
                    )
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = (
                        self.model.generation_config.mask_token_id
                    )

        return generation_config

    def _validate_generated_length(
        self, generation_config, input_ids_length, has_default_max_length
    ):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if (
            has_default_max_length
            and generation_config.max_new_tokens is None
            and generation_config.max_length == 20
        ):
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    @torch.no_grad()
    def diffusion_generate(
        self,
        input_ids: torch.LongTensor,
        generation_config: Optional[DreamGenerationConfig] = None,
        reward_fn: Optional[Callable] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # 2. Define model inputs
        assert input_ids is not None
        attention_mask = kwargs.pop("attention_mask", None)
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop(
            "generation_tokens_hook_func", lambda step, x, logits: x
        )
        generation_logits_hook_func = kwargs.pop(
            "generation_logits_hook_func", lambda step, x, logits: logits
        )

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = (
            kwargs.get("max_length") is None
            and generation_config.max_length is not None
        )
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(
            generation_config, input_ids_length, has_default_max_length
        )

        if (
            hasattr(generation_config, "pad_token_id")
            and torch.any(input_ids == generation_config.pad_token_id)
            and attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        result = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            reward_fn=reward_fn,
            generation_tokens_hook_func=generation_tokens_hook_func,
            generation_logits_hook_func=generation_logits_hook_func,
        )
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        reward_fn: Optional[Callable] = None,
        generation_tokens_hook_func=None,
        generation_logits_hook_func=None,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        # eps = generation_config.eps
        eps = 1e-12
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(
                attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0
            )
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, 1e-10, steps + 1, device=x.device)
        masked_seq_len = (x == mask_token_id).sum().item()
        x = generation_tokens_hook_func(None, x, None)
        i = 0
        while i < steps:
            mask_index = x == mask_token_id
            #### this is denoiser_fn step ###
            #################################
            logits = self.model(x, attention_mask, tok_idx).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            logits = generation_logits_hook_func(i, x, logits)
            #################################
            # NOTE: This is where we can evaluate the reward of the generated tokens
            # this would greatly slow down the generation speed
            # as we need to evaluate the reward with execution
            # NOTE: or we can evaluate the reward with larger strides
            # if reward_fn is not None:
            #     _x0 = torch.argmax(logits, dim=-1)
            #     # or call sample_tokens()
            #     intermediate_x0 = torch.where(mask_index, _x0, x)
            #     reward = reward_fn(intermediate_x0)
            # else:
            #     reward = None
            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]

            if alg == "origin":
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = (
                    torch.zeros_like(
                        x[mask_index], device=self.device, dtype=torch.long
                    )
                    + mask_token_id
                )
                transfer_index_t_s = (
                    torch.rand(*x0.shape, device=self.device) < p_transfer
                )
                _, x0[transfer_index_t_s] = sample_tokens(
                    mask_logits[transfer_index_t_s],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                x[mask_index] = x0.clone()
            elif alg.startswith("transition"):
                transition_step = int(alg.split("_")[1])
                confidence, x0 = sample_tokens(
                    mask_logits, temperature=temperature, top_p=top_p, top_k=top_k
                )
                if i < transition_step:
                    number_transfer_tokens = 1
                else:
                    # at 0 <= i <= transition_step, the number of transfer tokens is 1
                    # at i = transition_step, the number of transfer tokens is transition_step
                    # at i = steps, the number of transfer tokens is seq_len
                    # a * transition_step + b = transition_step
                    # a * steps + b = seq_len
                    denoised_tokens = int(
                        (
                            (masked_seq_len - transition_step) * (i + 1)
                            + transition_step * (steps - masked_seq_len)
                        )
                        / (steps - transition_step)
                    )
                    already_denoised_tokens = masked_seq_len - mask_index.sum().item()
                    number_transfer_tokens = denoised_tokens - already_denoised_tokens

                full_confidence = torch.full_like(
                    x, -torch.inf, device=self.device, dtype=logits.dtype
                )
                full_confidence[mask_index] = confidence
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(
                            full_confidence, number_transfer_tokens
                        )
                    else:
                        full_confidence = full_confidence / alg_temp
                        full_confidence = F.softmax(full_confidence, dim=-1)
                        transfer_index = torch.multinomial(
                            full_confidence, num_samples=number_transfer_tokens
                        )
                    x_ = (
                        torch.zeros_like(x, device=self.device, dtype=torch.long)
                        + mask_token_id
                    )
                    x_[mask_index] = x0.clone()
                    row_indices = (
                        torch.arange(x.size(0), device=self.device)
                        .unsqueeze(1)
                        .expand_as(transfer_index)
                    )
                    x[row_indices, transfer_index] = x_[row_indices, transfer_index]
            else:
                if alg == "maskgit_plus":
                    confidence, x0 = sample_tokens(
                        mask_logits, temperature=temperature, top_p=top_p, top_k=top_k
                    )
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")
                # NOTE: hack for debugging
                #################################
                # num_mask_token = mask_index.sum() / mask_index.shape[0]
                # number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
                # instead of using the number of mask tokens, we use 1, which is more stable
                number_transfer_tokens = 1
                #################################

                full_confidence = torch.full_like(
                    x, -torch.inf, device=self.device, dtype=logits.dtype
                )
                full_confidence[mask_index] = confidence
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(
                            full_confidence, number_transfer_tokens
                        )
                    else:
                        full_confidence = full_confidence / alg_temp
                        full_confidence = F.softmax(full_confidence, dim=-1)
                        transfer_index = torch.multinomial(
                            full_confidence, num_samples=number_transfer_tokens
                        )
                    x_ = (
                        torch.zeros_like(x, device=self.device, dtype=torch.long)
                        + mask_token_id
                    )
                    x_[mask_index] = x0.clone()
                    row_indices = (
                        torch.arange(x.size(0), device=self.device)
                        .unsqueeze(1)
                        .expand_as(transfer_index)
                    )
                    x[row_indices, transfer_index] = x_[row_indices, transfer_index]

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)
            eos_mask = ((x == generation_config.eos_token_id).cumsum(dim=-1) > 0).to(
                torch.bool
            )
            mask_before_casting = x == mask_token_id
            x[eos_mask] = generation_config.eos_token_id
            increments = (mask_before_casting & eos_mask).sum().item()
            assert x.shape[0] == 1, "batch size must be 1"
            i += increments
            i += 1

            if histories is not None:
                histories.append(x.clone())

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                num_steps=i,
                history=histories,
            )
        else:
            return x

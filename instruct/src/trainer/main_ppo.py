import os

import hydra
import ray

from src.trainer.ppo.ray_trainer import RayPPOTrainer


def get_custom_reward_fn(config):
    import importlib.util
    import os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")

    if not hasattr(module, function_name):
        raise AttributeError(
            f"Reward function '{function_name}' not found in '{file_path}'."
        )

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    return getattr(module, function_name)


@hydra.main(config_path="config", config_name="grpo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get(
        "CUDA_VISIBLE_DEVICES", ""
    )
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                }
            },
            dashboard_host="0.0.0.0",
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:

    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(
            OmegaConf.to_container(config, resolve=True)
        )  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_tokenizer

        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            from verl.single_controller.ray import RayWorkerGroup

            from src.workers.fsdp_workers import ActorRolloutRefWorker

            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError(
                f"Actor strategy {config.actor_rollout_ref.actor.strategy} is not supported"
            )

        from src.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        if config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # we use a naive reward manager for now
        reward_manager_name = config.reward_model.get("reward_manager", "ctrl")
        if reward_manager_name == "ctrl":
            from src.workers.reward_manager import AsyncRewardManager

            reward_manager_cls = AsyncRewardManager
        else:
            raise NotImplementedError(
                f"Reward manager {reward_manager_name} is not supported"
            )

        metadata_dir = None
        if config.reward_model.get("save_metadata", False):
            metadata_dir = config.trainer.default_local_dir
        reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=0,
            force_instruct_following=config.reward_model.get(
                "force_instruct_following", False
            ),
            metadata_dir=metadata_dir,
        )

        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=1,
            force_instruct_following=config.reward_model.get(
                "force_instruct_following", False
            ),
            metadata_dir=metadata_dir,
        )

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()

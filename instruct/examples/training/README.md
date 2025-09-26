# Training

## Prerequisites
Before training, ensure your data is prepared using the scripts in the [data preparation](../data_preparation) directory.

Run the following scripts for installation:
```bash
pip install -r requirements.txt
```

## Supervised Fine-tuning
To perform SFT on a single node with 8 GPUs, simply run:
```bash
bash examples/training/run_sft.sh
```

## Reinforcement Learning from Verifiable Sandbox Rewards
We utilize [SandboxFusion](https://github.com/bytedance/SandboxFusion) to set up a secure Python sandbox environment. This involves launching a multi-replica Docker service (using [Docker Swarm](https://docs.docker.com/engine/swarm/)) that exposes an entrypoint for verifying the correctness of generated code against unit tests.

For detailed SandboxFusion usage and setup, refer to their [documentation](https://bytedance.github.io/SandboxFusion/).

To start the SandboxFusion service, run:

```bash
bash examples/training/launch_sandbox.sh
```

Then, launch your RL training on [Dream-Coder-RL-17k](https://huggingface.co/datasets/Dream-org/Dream-Coder-RL-17k) with:

```bash
bash examples/training/run_rl.sh $CKPT_DIR $SANDBOX_FUSION_ENDPOINT
```

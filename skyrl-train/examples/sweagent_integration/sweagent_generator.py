import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from difflib import SequenceMatcher

from omegaconf import DictConfig

from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import ConversationType


def _get_patch_from_dir(output_dir: str) -> str:
    try:
        patch_file = next(Path(output_dir).rglob("*.patch"))
        with open(patch_file, "r") as f:
            return f.read()
    except StopIteration:
        return ""


def _seq_reward(reference_patch: str, pred_patch: str) -> float:
    diff = SequenceMatcher(None, reference_patch.splitlines(keepends=True), pred_patch.splitlines(keepends=True))
    return diff.ratio()


class SWEAgentGenerator(GeneratorInterface):
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer,
        inference_engine_client: InferenceEngineClient,
    ) -> None:
        # cfg is the full Hydra config
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.inference_engine_client = inference_engine_client

        # OpenAI-compatible endpoint exposed by InferenceEngineClient
        assert (
            cfg.generator.enable_http_endpoint
        ), "Set generator.enable_http_endpoint=true to expose the current policy via HTTP"
        self.model_name: str = cfg.trainer.policy.model.path
        self.api_base: str = f"http://{cfg.generator.http_endpoint_host}:{cfg.generator.http_endpoint_port}/v1"

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        prompts: List[ConversationType] = input_batch["prompts"]
        env_extras: Optional[List[Dict[str, Any]]] = input_batch.get("env_extras")
        if env_extras is None:
            env_extras = [{} for _ in prompts]

        # Expect extras per instance to contain required fields for sweagent
        # Required keys per instance:
        # - problem_statement: str
        # - instance_id: str
        # - base_commit: str
        # - patch: str (reference)

        responses: List[List[int]] = []
        prompt_token_ids: List[List[int]] = []
        loss_masks: List[List[int]] = []
        stop_reasons: List[str] = []
        rewards: List[float] = []

        # Resolve repo root and absolute config path for sweagent
        repo_root = os.environ.get("SKYRL_REPO_ROOT", "/mnt/efs/mohitraghavendra/src/SkyRL/skyrl-train")
        abs_config_path = os.path.join(repo_root, "examples/sweagent_integration/config/default_config.yaml")

        for idx, (prompt_messages, extras) in enumerate(zip(prompts, env_extras)):
            problem_statement: str = extras.get("problem_statement", "")
            instance_id: str = extras.get("instance_id", f"instance_{idx}")
            base_commit: str = extras.get("base_commit", "")
            reference_patch: str = extras.get("patch", "")

            # Build output dir per instance
            output_dir = (
                Path(self.cfg.trainer.export_path)
                / "sweagent_results"
                / "skyrl_integration"
                / instance_id.replace("/", "-")
            )
            output_dir_str = str(output_dir)
            os.makedirs(output_dir_str, exist_ok=True)

            # Build sweagent command using current SkyRL policy endpoint
            command = [
                sys.executable,
                "-m",
                "sweagent.run.run",
                "run",
                "--config", abs_config_path,
                "--output_dir", output_dir_str,
                "--env.deployment.type", "modal",
                "--env.deployment.image", extras.get("image_name", ""),
                "--env.repo.type", "preexisting",
                "--env.repo.repo_name", extras.get("repo_name", "testbed"), 
                "--env.repo.base_commit", base_commit,
                "--problem_statement.type", "text",
                "--problem_statement.text", problem_statement,  
                "--agent.model.name", self.model_name,
                "--agent.model.api_base", self.api_base,
                "--agent.model.per_instance_cost_limit", "0",
                "--agent.model.total_cost_limit", "0",
                "--agent.tools.total_execution_timeout", str(extras.get("total_execution_timeout", 60)),
                "--agent.model.api_key", os.environ.get("LITE_API_KEY", ""),
            ]

            # Ensure sweagent runs as if from repo root and can import local modules if needed
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{repo_root}:" + env.get("PYTHONPATH", "")
            subprocess.run(command, check=True, cwd=repo_root, env=env)

            # Read .patch and compute reward
            pred_patch = _get_patch_from_dir(output_dir_str)
            reward = _seq_reward(reference_patch, pred_patch)

            # Tokenize prompt and response text for training
            prompt_ids = self.tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                add_special_tokens=False,
                return_dict=True,
                tokenize=True,
            )["input_ids"]

            # We feed the patch text as the model response
            # If empty, give an empty response
            # TODO: We should calculate loss for the trajectory as well, not just the final patch

            response_text = pred_patch if pred_patch else ""
            response_ids = self.tokenizer(
                response_text,
                add_special_tokens=False,
                return_tensors=None,
            )["input_ids"]

            # Simple full mask for response tokens
            # TODO: We should calculate loss for the trajectory as well, not just the final patch
            loss_mask = [1] * len(response_ids)

            prompt_token_ids.append(prompt_ids)
            responses.append(response_ids)
            loss_masks.append(loss_mask)
            rewards.append(reward)
            stop_reasons.append("stop")

        return {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": None,
            "rollout_logprobs": None,
        }



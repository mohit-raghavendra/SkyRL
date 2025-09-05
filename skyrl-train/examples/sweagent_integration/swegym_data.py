import argparse
import os
from typing import Dict, Any

import datasets


def get_image_name(instance_id: str) -> str:
    idx = instance_id.replace("__", "_s_")
    return "xingyaoww/sweb.eval.x86_64." + idx + ":latest"


def build_prompt(problem_statement: str):
    # Single user message conversation
    return [
        {
            "role": "user",
            "content": problem_statement,
        }
    ]


def map_row(example: Dict[str, Any]) -> Dict[str, Any]:
    instance_id = example.get("instance_id", "")
    problem_statement = example.get("problem_statement", "")
    base_commit = example.get("base_commit", "")
    reference_patch = example.get("patch", "")

    image_name = get_image_name(instance_id) if instance_id else ""

    return {
        "data_source": "SWE-Gym/SWE-Gym",
        "prompt": build_prompt(problem_statement),
        # env_class can be omitted; generator ignores it
        # extras for SWEAgentGenerator
        "problem_statement": problem_statement,
        "instance_id": instance_id,
        "base_commit": base_commit,
        "patch": reference_patch,
        "image_name": image_name,
        # optional defaults
        "repo_name": example.get("repo_name", "testbed"),
        "total_execution_timeout": example.get("total_execution_timeout", 60),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess SWE-Gym into SkyRL parquet format")
    parser.add_argument("--output_dir", default="~/data/swegym")

    args = parser.parse_args()
    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    data_source = "SWE-Gym/SWE-Gym"
    ds = datasets.load_dataset(data_source)

    assert "train" in ds, "Expected a 'train' split in SWE-Gym/SWE-Gym"

    train_ds = ds["train"]
    # If no validation split, create one with 50 problems from train
    if "validation" in ds:
        val_ds = ds["validation"]
    else:
        assert len(train_ds) >= 50, "Need at least 50 training samples to create validation split"
        split = train_ds.train_test_split(test_size=50, seed=42, shuffle=True)
        train_ds = split["train"]
        val_ds = split["test"]

    train_out = train_ds.map(function=map_row).select(range(100))
    val_out = val_ds.map(function=map_row)

    train_path = os.path.join(args.output_dir, "train.parquet")
    train_subset_path = os.path.join(args.output_dir, "train_subset.parquet")
    val_path = os.path.join(args.output_dir, "validation.parquet")
    train_out.to_parquet(train_path)
    val_out.to_parquet(val_path)
    train_out.to_parquet(train_subset_path)
    print(f"Wrote train -> {train_path} ({len(train_out)} rows)")
    print(f"Wrote train subset -> {train_subset_path} ({len(train_out)} rows)")
    print(f"Wrote validation -> {val_path} ({len(val_out)} rows)")



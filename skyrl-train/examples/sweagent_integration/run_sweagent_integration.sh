set -x

export VLLM_USE_FLASHINFER_SAMPLER=0

DATA_DIR="$HOME/data/swegym"
NUM_GPUS=4
TP_SIZE=1
NUM_INFERENCE_ENGINES=1
LOGGER=wandb  # change to "wandb" to log to WandB

INFERENCE_BACKEND="vllm"


# Example run: enable HTTP endpoint so sweagent can call current policy
uv sync --extra vllm
source .venv/bin/activate

python -m examples.sweagent_integration.main \
  data.train_data="['$DATA_DIR/train_subset.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-0.6B" \
  trainer.placement.colocate_all=false \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.epochs=20 \
  trainer.eval_batch_size=2 \
  trainer.eval_before_train=true \
  trainer.eval_interval=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=2 \
  trainer.policy_mini_batch_size=1 \
  trainer.logger="$LOGGER" \
  trainer.project_name="sweagent_integration" \
  trainer.run_name="sweagent_integration_test" \
  trainer.ckpt_path="$HOME/ckpts/sweagent_integration/Qwen3-0-6B_ckpt" \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.enable_http_endpoint=true \
  generator.async_engine=true \
  generator.batched=true \
  generator.http_endpoint_host=127.0.0.1 \
  generator.http_endpoint_port=8085 \
  generator.gpu_memory_utilization=0.6 \
  generator.n_samples_per_prompt=4 \


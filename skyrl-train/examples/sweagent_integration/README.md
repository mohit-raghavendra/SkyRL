# SWEAgent Integration Example

This example uses a custom Generator that shells out to sweagent, while routing the LLM calls to the live SkyRL policy via the built-in OpenAI-compatible HTTP endpoint.

Quick start:

1. Ensure sweagent CLI is available in PATH.
2. Run the example:

```bash
bash skyrl-train/examples/sweagent_integration/run_sweagent_integration.sh
```

Key notes:
- Set generator.enable_http_endpoint=true so the current SkyRL policy is served at http://$host:$port/v1.
- The SWEAgentGenerator reads model name and endpoint from the current config: cfg.trainer.policy.model.path, cfg.generator.http_endpoint_host, cfg.generator.http_endpoint_port.
- Reward uses simple sequence similarity (difflib) between the env_extras.patch (reference) and the produced .patch from sweagent.
- Dataset rows must provide env_extras fields per instance:
  - problem_statement: str
  - instance_id: str
  - base_commit: str
  - patch: str (reference)
  - Optional: image_name, repo_name, total_execution_timeout

Outputs:
- Per-instance sweagent outputs are stored under ${trainer.export_path}/sweagent_results/skyrl_integration/{instance_id}.



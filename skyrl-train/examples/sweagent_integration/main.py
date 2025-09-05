from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils import initialize_ray
from omegaconf import DictConfig
import ray
import hydra


class SWEAgentExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        from .sweagent_generator import SWEAgentGenerator

        return SWEAgentGenerator(cfg=cfg,
                                 tokenizer=tokenizer,
                                 inference_engine_client=inference_engine_client)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    exp = SWEAgentExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()



import hydra
from omegaconf import DictConfig
from rich import print
import rootutils

rootutils.setup_root(__file__, ".project-root", pythonpath="True")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(conf: DictConfig):
    print(
        f'Preparing to execute the following actions: {" | ".join(conf.actions.keys())}'
    )
    print("Executor starting ...")
    for action_name, action_value in conf.actions.items():
        print(f"Executing action: {action_name} ...")
        hydra.utils.call(
            action_value,
        )
        print(f"Execution of action: {action_name} finished.")
    print("Execution of all actions finished.")


if __name__ == "__main__":
    main()

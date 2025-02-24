"""Runs training and evaluation loop for the Z-Bot."""

import argparse

from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.open_duck_mini_v2 import joystick


class OpenDuckMiniV2Runner(BaseRunner):

    def __init__(self, args):
        # TODO select the env depending on the task etc
        super().__init__(args)
        self.env_config = joystick.default_config()
        self.env = joystick.Joystick(task=args.task)
        self.eval_env = joystick.Joystick(task=args.task)
        self.randomizer = randomize.domain_randomize
        self.action_size = self.env.action_size
        self.obs_size = int(self.env.observation_size["state"][0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Open Duck Mini Runner Script")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Where to save the checkpoints",
    )
    parser.add_argument("--task", type=str, default="flat_terrain", help="Task to run")
    # parser.add_argument(
    #     "--debug", action="store_true", help="Run in debug mode with minimal parameters"
    # )
    args = parser.parse_args()

    runner = OpenDuckMiniV2Runner(args)

    runner.train()


if __name__ == "__main__":
    main()

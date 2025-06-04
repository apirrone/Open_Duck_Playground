import argparse

from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.new_hopejr import joystick, joystick_retarget


class HopeJRRunner(BaseRunner):

    def __init__(self, args):
        # TODO select the env depending on the task etc
        super().__init__(args)
        available_envs = {
            "joystick": (joystick, joystick.Joystick),
            "joystick_retarget": (joystick_retarget, joystick_retarget.JoystickRetarget),
        }
        if args.env not in available_envs:
            raise ValueError(f"Unknown env {args.env}")

        self.env_file = available_envs[args.env]

        self.env_config = self.env_file[0].default_config()
        self.env = self.env_file[1](task=args.task)
        self.eval_env = self.env_file[1](task=args.task)
        self.randomizer = randomize.domain_randomize
        self.action_size = self.env.action_size
        self.obs_size = int(
            self.env.observation_size["state"][0]
        )  # 0: state 1: privileged_state
        print(f"Observation size: {self.obs_size}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HopeJR Runner Script")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Where to save the checkpoints",
    )
    parser.add_argument("--task", type=str, default="flat_terrain", help="Task to run")
    parser.add_argument("--num_timesteps", type=int, default=150000000)
    parser.add_argument("--env", type=str, default="joystick", help="env")
    # parser.add_argument(
    #     "--debug", action="store_true", help="Run in debug mode with minimal parameters"
    # )
    args = parser.parse_args()

    runner = HopeJRRunner(args)

    runner.train()


if __name__ == "__main__":
    main()

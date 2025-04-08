"""
Defines a common runner between the different robots.
Inspired from https://github.com/kscalelabs/mujoco_playground/blob/master/playground/common/runner.py
"""

from pathlib import Path
from abc import ABC
import argparse
import functools
from datetime import datetime
import time
from flax.training import orbax_utils
from tensorboardX import SummaryWriter

import os
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp
import jax

from playground.common.jax_config import configure_jax_memory
from playground.common.export_onnx import export_onnx


class BaseRunner(ABC):
    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the Runner class.

        Args:
            args (argparse.Namespace): Command line arguments.
        """
        # Configure JAX memory usage first thing
        # Use 80% of GPU memory for JAX, leaving 20% for other operations
        configure_jax_memory(memory_fraction=0.8)
        
        # Configure logging to suppress the multi-process warning
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.ERROR)
        
        self.args = args
        self.output_dir = args.output_dir
        self.output_dir = Path.cwd() / Path(self.output_dir)

        self.env_config = None
        self.env = None
        self.eval_env = None
        self.randomizer = None
        self.writer = SummaryWriter(log_dir=self.output_dir)
        self.action_size = None
        self.obs_size = None
        self.num_timesteps = args.num_timesteps
        self.restore_checkpoint_path = None
        
        # Add attributes to control export frequencies
        self.onnx_export_frequency = getattr(args, 'onnx_export_frequency', 10)
        self.export_onnx_enabled = getattr(args, 'export_onnx', True)
        self.checkpoint_frequency = getattr(args, 'checkpoint_frequency', 1)  # Default: save every evaluation
        self.eval_step_interval = getattr(args, 'eval_step_interval', 1)  # Default: evaluate every 1M steps
        
        # Track timing between evaluations
        self.last_eval_time = time.time()
        self.last_eval_step = 0
        
        # CACHE STUFF
        os.makedirs(".tmp", exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", ".tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update(
            "jax_persistent_cache_enable_xla_caches",
            "xla_gpu_per_fusion_autotune_cache_dir",
        )  
        os.environ["JAX_COMPILATION_CACHE_DIR"] = ".tmp/jax_cache"

    def progress_callback(self, num_steps: int, metrics: dict) -> None:
        """Log training metrics to TensorBoard and console at each callback step.
        
        Args:
            num_steps: Current training step number
            metrics: Dictionary containing various training and evaluation metrics
        """
        # Calculate time since last evaluation
        current_time = time.time()
        time_elapsed = current_time - self.last_eval_time
        steps_done = num_steps - self.last_eval_step
        
        # Calculate steps per second
        steps_per_second = steps_done / time_elapsed if time_elapsed > 0 else 0
        time_per_1m_steps = 1_000_000 / steps_per_second if steps_per_second > 0 else 0
        
        # Update for next evaluation
        self.last_eval_time = current_time
        self.last_eval_step = num_steps
        
        # Log all metrics to TensorBoard
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(metric_name, metric_value, num_steps)
        
        # Add timing metrics
        self.writer.add_scalar("timing/steps_per_second", steps_per_second, num_steps)
        self.writer.add_scalar("timing/minutes_per_million_steps", time_per_1m_steps / 60, num_steps)

        # Print key metrics to console for monitoring
        print("-----------")
        print(
            f"STEP: {num_steps} | "
            f"Reward: {metrics['eval/episode_reward']:.4f} | "
            f"Reward std: {metrics['eval/episode_reward_std']:.4f} | "
            f"Time for {steps_done:,} steps: {time_elapsed:.2f}s | "
            f"Steps/sec: {steps_per_second:.2f} | "
            f"Time per 1M steps: {time_per_1m_steps/60:.2f} minutes"
        )
        print("-----------")

    def save_checkpoint(self, params, current_step):
        """Save model parameters to checkpoint file.
        
        Args:
            params: Model parameters to save
            current_step: Current training step number
        
        Returns:
            path: Path where checkpoint was saved
        """
        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        checkpoint_path = f"{self.output_dir}/{timestamp}_{current_step}"
        
        print(f"Saving checkpoint (step: {current_step}): {checkpoint_path}")
        orbax_checkpointer.save(checkpoint_path, params, force=True, save_args=save_args)
        
        return checkpoint_path
    
    def export_model_to_onnx(self, params, checkpoint_path, current_step):
        """Export trained model to ONNX format.
        
        Args:
            params: Model parameters to export
            checkpoint_path: Path of the saved checkpoint
            current_step: Current training step number
        """
        if not self.export_onnx_enabled:
            print("ONNX export disabled, skipping...")
            return
            
        onnx_export_path = f"{checkpoint_path}.onnx"
        try:
            print(f"Attempting to export model to ONNX: {onnx_export_path}")
            export_onnx(
                params,
                self.action_size,
                self.ppo_params,
                self.obs_size,
                output_path=onnx_export_path
            )
            print(f"Successfully exported model to ONNX: {onnx_export_path}")
        except Exception as e:
            print(f"WARNING: Failed to export model to ONNX format: {e}")
            print("Continuing training without ONNX export...")

    def policy_params_fn(self, current_step, make_policy, params):
        """Callback for saving model checkpoints and exporting to ONNX.
        
        Args:
            current_step: Current training step
            make_policy: Function to create policy from parameters
            params: Current model parameters
        """
        # Calculate which evaluation number this is based on step count
        eval_num = self.ppo_training_params["num_evals"] * current_step // self.num_timesteps
        
        # Only save checkpoint at specified frequency
        if eval_num % self.checkpoint_frequency == 0:
            print(f"Evaluation {eval_num} is a multiple of {self.checkpoint_frequency}, saving checkpoint")
            checkpoint_path = self.save_checkpoint(params, current_step)
        else:
            print(f"Skipping checkpoint save for evaluation {eval_num} (not a multiple of {self.checkpoint_frequency})")
            # Create a temporary path for ONNX export if needed
            timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
            checkpoint_path = f"{self.output_dir}/{timestamp}_{current_step}"
        
        # Only export to ONNX every N evaluations
        if eval_num % self.onnx_export_frequency == 0 and self.export_onnx_enabled:
            print(f"Evaluation {eval_num} is a multiple of {self.onnx_export_frequency}, exporting ONNX model")
            self.export_model_to_onnx(params, checkpoint_path, current_step)
        else:
            print(f"Skipping ONNX export for evaluation {eval_num} (not a multiple of {self.onnx_export_frequency} or export disabled)")

    def configure_network_factory(self):
        """Configure the network factory for PPO based on parameters.
        
        Returns:
            Configured network factory function
        """
        if "network_factory" in self.ppo_params:
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks, 
                **self.ppo_params.network_factory
            )
            # Remove network_factory from training params to avoid duplication
            del self.ppo_training_params["network_factory"]
        else:
            network_factory = ppo_networks.make_ppo_networks
            
        return network_factory

    def train(self) -> None:
        """Train the policy using PPO algorithm.
        
        Sets up environment, configures PPO parameters, and runs training.
        The trained policy parameters are returned.
        """
        # Initialize PPO parameters based on environment
        # TODO: Replace hardcoded environment name with parameter
        env_name = "BerkeleyHumanoidJoystickFlatTerrain"
        self.ppo_params = locomotion_params.brax_ppo_config(env_name)
        self.ppo_training_params = dict(self.ppo_params)
        
        # Configure network and training parameters
        network_factory = self.configure_network_factory()
        self.ppo_training_params["num_timesteps"] = self.num_timesteps
        
        # Set evaluation frequency to every N million steps based on user parameter
        # num_evals determines how many times metrics are logged during training
        # We add 1 to account for the initial evaluation at step 0
        eval_interval_steps = self.eval_step_interval * 1_000_000
        self.ppo_training_params["num_evals"] = (self.num_timesteps // eval_interval_steps) + 1
        
        print(f"PPO params: {self.ppo_training_params}")
        print(f"Logging metrics every ~{self.eval_step_interval}M steps ({self.ppo_training_params['num_evals']} evaluations)")

        # Configure the training function
        train_fn = functools.partial(
            ppo.train,
            **self.ppo_training_params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback,
            policy_params_fn=self.policy_params_fn,
            restore_checkpoint_path=self.restore_checkpoint_path,
        )

        # Run the training
        _, params, _ = train_fn(
            environment=self.env,
            eval_env=self.eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
        
        return params

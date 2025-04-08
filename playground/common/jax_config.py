"""Configure JAX memory allocation to leave room for TensorFlow operations."""

import os
import jax
import logging

def configure_jax_memory(memory_fraction=0.8):
    """
    Configure JAX to limit GPU memory usage to a fraction of available memory.
    
    Args:
        memory_fraction: Fraction of GPU memory that JAX is allowed to use (0.0 to 1.0)
    """
    # Set environment variable to limit JAX GPU memory usage
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    # Setup JAX to grow memory as needed instead of allocating all at once
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    
    # Configure JAX logging
    logging.info(f"Configured JAX to use up to {memory_fraction*100}% of available GPU memory")
    
    # Verify the settings took effect
    devices = jax.devices()
    logging.info(f"JAX using devices: {devices}")
    
    return devices

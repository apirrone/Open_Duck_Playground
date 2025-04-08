import tensorflow as tf
from tensorflow.keras import layers
import tf2onnx
import numpy as np
import os
import contextlib

@contextlib.contextmanager
def tensorflow_cpu_only_context():
    """Temporarily force TensorFlow to use CPU only."""
    # Save original environment variable
    original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    
    # Set environment variable to hide GPU from TensorFlow
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Clear any existing TensorFlow sessions/memory
    tf.keras.backend.clear_session()
    
    try:
        yield
    finally:
        # Restore original environment variable
        if original_cuda_devices is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        
        # Clean up again
        tf.keras.backend.clear_session()

def export_onnx(
    params, act_size, ppo_params, obs_size, output_path="ONNX.onnx"
):
    print(" === EXPORT ONNX === ")
    
    # Cleanup before starting and force CPU-only for TensorFlow
    with tensorflow_cpu_only_context():
        print("TensorFlow configured to use CPU only for ONNX export")
        
        # Convert JAX parameters to NumPy arrays first
        # This helps prevent device transfers that could trigger OOM
        mean = np.array(params[0].mean["state"])
        std = np.array(params[0].std["state"])
        policy_params = {
            layer_name: {
                param_name: np.array(param_val) 
                for param_name, param_val in layer_params.items()
            }
            for layer_name, layer_params in params[1].policy["params"].items()
        }
        
        # Define the MLP model
        class MLP(tf.keras.Model):
            def __init__(
                self,
                layer_sizes,
                activation=tf.nn.relu,
                kernel_init="lecun_uniform",
                activate_final=False,
                bias=True,
                layer_norm=False,
                mean_std=None,
            ):
                super().__init__()

                self.layer_sizes = layer_sizes
                self.activation = activation
                self.kernel_init = kernel_init
                self.activate_final = activate_final
                self.bias = bias
                self.layer_norm = layer_norm

                if mean_std is not None:
                    self.mean = tf.Variable(mean_std[0], trainable=False, dtype=tf.float32)
                    self.std = tf.Variable(mean_std[1], trainable=False, dtype=tf.float32)
                else:
                    self.mean = None
                    self.std = None

                self.mlp_block = tf.keras.Sequential(name="MLP_0")
                for i, size in enumerate(self.layer_sizes):
                    dense_layer = layers.Dense(
                        size,
                        activation=self.activation,
                        kernel_initializer=self.kernel_init,
                        name=f"hidden_{i}",
                        use_bias=self.bias,
                    )
                    self.mlp_block.add(dense_layer)
                    if self.layer_norm:
                        self.mlp_block.add(
                            layers.LayerNormalization(name=f"layer_norm_{i}")
                        )
                if not self.activate_final and self.mlp_block.layers:
                    if (
                        hasattr(self.mlp_block.layers[-1], "activation")
                        and self.mlp_block.layers[-1].activation is not None
                    ):
                        self.mlp_block.layers[-1].activation = None

                self.submodules = [self.mlp_block]

            def call(self, inputs):
                if isinstance(inputs, list):
                    inputs = inputs[0]
                if self.mean is not None and self.std is not None:
                    print(self.mean.shape, self.std.shape)
                    inputs = (inputs - self.mean) / self.std
                logits = self.mlp_block(inputs)
                loc, _ = tf.split(logits, 2, axis=-1)
                return tf.tanh(loc)

        def make_policy_network(
            param_size,
            mean_std,
            hidden_layer_sizes=[256, 256],
            activation=tf.nn.relu,
            kernel_init="lecun_uniform",
            layer_norm=False,
        ):
            policy_network = MLP(
                layer_sizes=list(hidden_layer_sizes) + [param_size],
                activation=activation,
                kernel_init=kernel_init,
                layer_norm=layer_norm,
                mean_std=mean_std,
            )
            return policy_network

        # Create tensors on CPU
        mean_std = (tf.convert_to_tensor(mean), tf.convert_to_tensor(std))

        tf_policy_network = make_policy_network(
            param_size=act_size * 2,
            mean_std=mean_std,
            hidden_layer_sizes=ppo_params.network_factory.policy_hidden_layer_sizes,
            activation=tf.nn.swish,
        )

        example_input = tf.zeros((1, obs_size))
        example_output = tf_policy_network(example_input)
        print(example_output.shape)

        def transfer_weights(numpy_params, tf_model):
            """
            Transfer weights from a NumPy parameter dictionary to the TensorFlow model.

            Parameters:
            - numpy_params: dict
            Nested dictionary with structure {block_name: {layer_name: {params}}}.
            For example:
            {
                'MLP_0': {
                'hidden_0': {'kernel': np.ndarray, 'bias': np.ndarray},
                'hidden_1': {'kernel': np.ndarray, 'bias': np.ndarray},
                'hidden_2': {'kernel': np.ndarray, 'bias': np.ndarray},
                }
            }

            - tf_model: tf.keras.Model
            An instance of the adapted VisionMLP model containing named submodules and layers.
            """
            for layer_name, layer_params in numpy_params.items():
                try:
                    tf_layer = tf_model.get_layer("MLP_0").get_layer(name=layer_name)
                except ValueError:
                    print(f"Layer {layer_name} not found in TensorFlow model.")
                    continue
                if isinstance(tf_layer, tf.keras.layers.Dense):
                    kernel = np.array(layer_params["kernel"])
                    bias = np.array(layer_params["bias"])
                    print(
                        f"Transferring Dense layer {layer_name}, kernel shape {kernel.shape}, bias shape {bias.shape}"
                    )
                    tf_layer.set_weights([kernel, bias])
                else:
                    print(f"Unhandled layer type in {layer_name}: {type(tf_layer)}")

            print("Weights transferred successfully.")

        # Transfer weights using NumPy arrays
        transfer_weights(policy_params, tf_policy_network)

        # Test the model with sample input
        test_input = [np.ones((1, obs_size), dtype=np.float32)]
        spec = [tf.TensorSpec(shape=(1, obs_size), dtype=tf.float32, name="obs")]
        tensorflow_pred = tf_policy_network(test_input)[0]
        print(f"Tensorflow prediction: {tensorflow_pred}")

        tf_policy_network.output_names = ["continuous_actions"]

        try:
            # Export to ONNX
            model_proto, _ = tf2onnx.convert.from_keras(
                tf_policy_network, input_signature=spec, opset=11, output_path=output_path
            )
            print(f"Successfully exported model to {output_path}")
            
            # For Antoine :)
            if output_path != "ONNX.onnx":
                model_proto, _ = tf2onnx.convert.from_keras(
                    tf_policy_network, input_signature=spec, opset=11, output_path="ONNX.onnx"
                )
                
            return True
        except Exception as e:
            print(f"ERROR during ONNX export: {e}")
            return False

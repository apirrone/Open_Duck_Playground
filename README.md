# Open Duck Playground

# Installation 

Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### macOS Requirements
For MuJoCo on macOS, you'll need to use `mjpython` instead of regular Python:

```bash
# Install MuJoCo
brew install mujoco

# Create conda environment
conda create -n open_duck_env python=3.11
conda activate open_duck_env

# Install dependencies
pip install -e .
```

# Training

If you want to use the [imitation reward](https://la.disneyresearch.com/wp-content/uploads/BD_X_paper.pdf), you can generate reference motion with [this repo](https://github.com/apirrone/Open_Duck_reference_motion_generator)

Then copy `polynomial_coefficients.pkl` in `playground/<robot>/data/`

You'll also have to set `USE_IMITATION_REWARD=True` in it's `joystick.py` file

Run: 

```bash
uv run playground/<robot>/runner.py 
```

## Tensorboard

```bash
uv run tensorboard --logdir=<yourlogdir>
```

# Inference 

Infer mujoco

(for now this is specific to open_duck_mini_v2)

```bash
uv run playground/open_duck_mini_v2/mujoco_infer.py -o <path_to_.onnx>
```

### Basic MuJoCo Inference
```bash
# Linux/Windows
uv run playground/open_duck_mini_v2/mujoco_infer.py -o <path_to_.onnx>

# macOS
mjpython -m playground.open_duck_mini_v2.mujoco_infer -o <path_to_.onnx>
```

### API Control (New!)
The robot can now be controlled via HTTP API and WebSocket:

```bash
# Run with API server (recommended)
mjpython -m playground.open_duck_mini_v2.mujoco_with_api -o <path_to_.onnx>
```

This starts both the MuJoCo viewer and API server on http://localhost:8000

#### Quick API Demo
```bash
# Move forward
curl -X POST http://localhost:8000/command/move/forward

# Turn left
curl -X POST http://localhost:8000/command/turn/left

# Stop
curl -X POST http://localhost:8000/command/stop

# Get status
curl http://localhost:8000/status | jq .

# Set precise velocities
curl -X POST http://localhost:8000/command/locomotion \
  -H "Content-Type: application/json" \
  -d '{"forward": 0.1, "lateral": 0.0, "angular": 0.5}'
```

See [API_CONTROL_README.md](API_CONTROL_README.md) for full API documentation.

### Voice Control (New!)
Control the robot using voice commands:

```bash
# Recommended: Use with language filtering
python playground/open_duck_mini_v2/run_voice_control.py --no-wake-word --filter-languages

# With wake word "duck duck"
python playground/open_duck_mini_v2/run_voice_control.py --filter-languages
```

Voice commands include: "go forward", "turn left", "stop", "look up", etc.
Use `--filter-languages` to prevent false triggers from non-English speech.

# Documentation

## Project structure : 

```
.
├── pyproject.toml
├── README.md
├── playground
│   ├── common
│   │   ├── export_onnx.py
│   │   ├── onnx_infer.py
│   │   ├── poly_reference_motion.py
│   │   ├── randomize.py
│   │   ├── rewards.py
│   │   └── runner.py
│   ├── open_duck_mini_v2
│   │   ├── base.py
│   │   ├── data
│   │   │   └── polynomial_coefficients.pkl
│   │   ├── joystick.py
│   │   ├── mujoco_infer.py
│   │   ├── constants.py
│   │   ├── runner.py
│   │   └── xmls
│   │       ├── assets
│   │       ├── open_duck_mini_v2_no_head.xml
│   │       ├── open_duck_mini_v2.xml
│   │       ├── scene_mjx_flat_terrain.xml
│   │       ├── scene_mjx_rough_terrain.xml
│   │       └── scene.xml
```

**FastAPI Integration**

The project now includes FastAPI integration for remote control:

```
playground/open_duck_mini_v2/
├── mujoco_infer.py         # Original inference script
├── mujoco_with_api.py      # Combined launcher (new)
├── api_server.py           # FastAPI server (new)
├── command_handler.py      # Command processing (new)
└── ... (other files)
```

## Adding a new robot

Create a new directory in `playground` named after `<your robot>`. You can copy the `open_duck_mini_v2` directory as a starting point.

You will need to:
- Edit `base.py`: Mainly renaming stuff to match you robot's name
- Edit `constants.py`: specify the names of some important geoms, sensors etc
  - In your `mjcf`, you'll probably have to add some sites, name some bodies/geoms and add the sensors. Look at how we did it for `open_duck_mini_v2`
- Add your `mjcf` assets in `xmls`. 
- Edit `joystick.py` : to choose the rewards you are interested in
  - Note: for now there is still some hard coded values etc. We'll improve things on the way
- Edit `runner.py`



# Notes

Inspired from https://github.com/kscalelabs/mujoco_playground


## Current win

```bash
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000
```
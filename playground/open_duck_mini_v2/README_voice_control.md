# Voice Control for Open Duck Robot

This module provides voice control capabilities for the Open Duck robot using RealtimeSTT with wake word detection.

## Features

- Wake word detection using OpenWakeWord (supports custom wake words)
- Natural language command recognition
- Real-time speech-to-text with low latency
- Support for multiple Whisper model sizes
- Full Apple Silicon (M1/M2/M3) compatibility

## Installation

The required dependencies have been added to `pyproject.toml`. Install them with:

```bash
pip install -e .
```

## Usage

### Basic Usage

Run the voice control with default wake word "duck duck":

```bash
python playground/open_duck_mini_v2/run_voice_control.py
```

### Custom Wake Word

You can use any custom wake word:

```bash
python playground/open_duck_mini_v2/run_voice_control.py --wake-word "hey robot"
```

### Command Line Options

```bash
python playground/open_duck_mini_v2/run_voice_control.py --help

Options:
  --api-url URL         API endpoint URL (default: http://localhost:8000)
  --model SIZE          Whisper model size: tiny, base, small, medium, large (default: tiny)
  --language CODE       Language code (default: en)
  --wake-word PHRASE    Wake word phrase (default: 'duck duck')
  --debug              Enable debug logging
```

### Example Commands

After saying your wake word, you can use these commands:

**Movement:**
- "go forward" / "move forward"
- "go backward" / "move back"
- "go left" / "move left"
- "go right" / "move right"

**Turning:**
- "turn left"
- "turn right"

**Speed Modifiers:**
- "go forward slowly"
- "turn left fast"
- "move backward quickly"

**Head Control:**
- "look up" / "head up"
- "look down" / "head down"
- "look left"
- "look right"

**Control Modes:**
- "switch to head mode" / "head mode"
- "switch to walk mode" / "locomotion mode"

**Stop Commands:**
- "stop"
- "emergency stop"

## Architecture

The voice control system consists of:

1. **voice_controller.py**: Core RealtimeSTT integration with OpenWakeWord
2. **voice_commands.py**: Natural language command parsing
3. **voice_config.py**: Configuration constants
4. **run_voice_control.py**: Main entry point

## Performance

On Apple Silicon (M1/M2/M3):
- Wake word detection: <200ms
- Command recognition: ~200-300ms (tiny model)
- Total latency: <500ms from speech to robot movement

## Wake Word Detection

This implementation uses OpenWakeWord, which:
- Supports custom wake words without requiring API keys
- Works natively on Apple Silicon
- Uses ONNX models for efficient inference
- Can be trained on custom wake words if needed

## Troubleshooting

### No Audio Input
- Check microphone permissions in System Settings > Privacy & Security > Microphone
- Verify audio device with: `python -m sounddevice`

### Model Download Issues
- Models are downloaded on first use
- Ensure sufficient disk space (~40MB for tiny model, ~50MB for OpenWakeWord models)
- First run may take longer due to model downloads

### API Connection Errors
- Verify the duck API is running: `python playground/open_duck_mini_v2/mujoco_with_api.py`
- Check the API URL matches your setup (default: http://localhost:8000)

### Wake Word Not Detected
- Speak clearly and ensure the microphone is working
- Try adjusting the microphone volume
- OpenWakeWord may need to download models on first use

## Development

To add new commands, edit the patterns in `voice_commands.py`:

```python
self.movement_patterns = {
    r'\bnew_pattern\b': ('move', 'direction'),
    # Add more patterns here
}
```

To train custom wake word models, refer to the OpenWakeWord documentation.
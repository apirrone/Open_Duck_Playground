# Voice Control for Open Duck Robot

This module provides voice control capabilities for the Open Duck robot using RealtimeSTT with wake word detection.

## Features

- Wake word detection using Picovoice Porcupine or OpenWakeWord
- Natural language command recognition
- Real-time speech-to-text with low latency
- Support for multiple Whisper model sizes
- Full Apple Silicon (M1/M2/M3) compatibility
- On-device processing (no audio sent to servers)

## Installation

The required dependencies have been added to `pyproject.toml`. Install them with:

```bash
pip install -e .
```

## Usage

### Basic Usage

#### Using Picovoice Porcupine (Recommended)

1. Get a free Picovoice access key:
   - Sign up at https://console.picovoice.ai/
   - Copy your access key

2. Run with built-in wake word "computer":
```bash
python playground/open_duck_mini_v2/run_voice_control.py --wake-word computer --porcupine-access-key YOUR_ACCESS_KEY
```

#### Using OpenWakeWord (Legacy)

Run the voice control with default wake word "duck duck":

```bash
python playground/open_duck_mini_v2/run_voice_control.py --wake-word-backend oww
```

### Custom Wake Word

#### Picovoice Porcupine

1. Create custom wake word at https://console.picovoice.ai/
2. Download the .ppn model file
3. Run with custom model:

```bash
python playground/open_duck_mini_v2/run_voice_control.py --porcupine-access-key YOUR_KEY --porcupine-keyword-path /path/to/your/model.ppn
```

#### OpenWakeWord (Legacy)

You can use any custom wake word:

```bash
python playground/open_duck_mini_v2/run_voice_control.py --wake-word "hey robot" --wake-word-backend oww
```

### Command Line Options

```bash
python playground/open_duck_mini_v2/run_voice_control.py --help

Options:
  --api-url URL                 API endpoint URL (default: http://localhost:8000)
  --model SIZE                  Whisper model size: tiny, base, small, medium, large (default: tiny)
  --language CODE               Language code (default: en)
  --wake-word PHRASE            Wake word phrase (default: 'duck duck')
  --wake-word-backend BACKEND   Wake word backend: porcupine, oww (default: porcupine)
  --porcupine-access-key KEY    Picovoice access key for Porcupine
  --porcupine-keyword-path PATH Path to custom .ppn model file
  --wake-word-sensitivity FLOAT Detection sensitivity 0.0-1.0 (default: 0.5)
  --no-wake-word                Run in always-listening mode
  --debug                       Enable debug logging
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

## Wake Word Detection

### Picovoice Porcupine (Recommended)

**Built-in Wake Words:**
- alexa
- americano
- blueberry
- bumblebee
- computer
- grapefruit
- grasshopper
- hey google
- hey siri
- jarvis
- ok google
- picovoice
- porcupine
- terminator

**Advantages:**
- Higher accuracy than OpenWakeWord
- Zero false positives in testing
- Custom wake words via web console
- All processing on-device (no audio sent to servers)
- Works great on macOS/Apple Silicon

### OpenWakeWord (Legacy)

This implementation uses OpenWakeWord, which:
- Supports custom wake words without requiring API keys
- Works natively on Apple Silicon
- Uses ONNX models for efficient inference
- Can be trained on custom wake words if needed

## Architecture

The voice control system consists of:

1. **voice_controller.py**: Core RealtimeSTT integration with Porcupine/OpenWakeWord
2. **porcupine_wake_word.py**: Picovoice Porcupine integration
3. **voice_commands.py**: Natural language command parsing
4. **voice_config.py**: Configuration constants
5. **run_voice_control.py**: Main entry point
6. **test_porcupine_wake_word.py**: Wake word testing utility

## Performance

On Apple Silicon (M1/M2/M3):
- Wake word detection: <200ms
- Command recognition: ~200-300ms (tiny model)
- Total latency: <500ms from speech to robot movement

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

**For Porcupine:**
- Ensure you have a valid access key
- Try a built-in wake word like "computer" or "jarvis"
- Adjust sensitivity with --wake-word-sensitivity (0.0-1.0)
- Test with: `python playground/open_duck_mini_v2/test_porcupine_wake_word.py --access-key YOUR_KEY`

**For OpenWakeWord:**
- Speak clearly and ensure the microphone is working
- Try adjusting the microphone volume
- OpenWakeWord may need to download models on first use
- Note: Custom wake words may not work reliably

## Development

To add new commands, edit the patterns in `voice_commands.py`:

```python
self.movement_patterns = {
    r'\bnew_pattern\b': ('move', 'direction'),
    # Add more patterns here
}
```

To train custom wake word models, refer to the OpenWakeWord documentation.
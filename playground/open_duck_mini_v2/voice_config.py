"""
Voice control configuration for Open Duck robot
"""

# API Configuration
DEFAULT_API_URL = "http://localhost:8000"
API_TIMEOUT = 5.0  # seconds

# Voice Recognition Configuration
DEFAULT_MODEL_SIZE = "tiny"  # Options: tiny, base, small, medium, large
DEFAULT_LANGUAGE = "en"
DEFAULT_WAKE_WORDS = ["duck duck"]

# Audio Configuration
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_CHUNK_SIZE = 1024

# Voice Activity Detection
VAD_SENSITIVITY = 0.4  # 0.0 to 1.0, higher is more sensitive
POST_SPEECH_SILENCE_DURATION = 0.4  # seconds
MIN_RECORDING_LENGTH = 0.5  # seconds
MIN_GAP_BETWEEN_RECORDINGS = 0.3  # seconds

# Real-time Transcription
ENABLE_REALTIME_TRANSCRIPTION = True
REALTIME_PROCESSING_PAUSE = 0.2  # seconds

# Command Recognition
COMMAND_CONFIDENCE_THRESHOLD = 0.7  # 0.0 to 1.0
MAX_COMMAND_LENGTH = 100  # characters

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Speed Modifiers
SPEED_SLOW = 0.3
SPEED_NORMAL = 0.5
SPEED_FAST = 0.8

# Head Movement Parameters
HEAD_PITCH_INCREMENT = 0.3
HEAD_YAW_INCREMENT = 0.5
HEAD_ROLL_INCREMENT = 0.2

# Retry Configuration
MAX_API_RETRIES = 3
RETRY_DELAY = 0.5  # seconds
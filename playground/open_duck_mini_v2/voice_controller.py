import os
import threading
import time
from typing import Optional, Callable, Dict, Any
from RealtimeSTT import AudioToTextRecorder
import logging

logger = logging.getLogger(__name__)


class DuckVoiceController:
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        model_size: str = "tiny",
        language: str = "en",
        wake_words: list = None,
        on_wake_word: Optional[Callable] = None,
        on_command: Optional[Callable[[str], None]] = None
    ):
        self.api_url = api_url
        self.model_size = model_size
        self.language = language
        self.wake_words = wake_words
        self.on_wake_word_callback = on_wake_word
        self.on_command_callback = on_command
        self.is_listening = False
        self.recorder = None
        self._init_recorder()
        
    def _init_recorder(self):
        try:
            # Base configuration
            config = {
                "model": self.model_size,
                "language": self.language,
                "silero_sensitivity": 0.4,
                "webrtc_sensitivity": 3,
                "post_speech_silence_duration": 0.4,
                "min_length_of_recording": 0.5,
                "min_gap_between_recordings": 0.3,
                "enable_realtime_transcription": True,
                "realtime_processing_pause": 0.2,
                "realtime_model_type": self.model_size,
                "on_realtime_transcription_update": self._on_transcription_update,
            }
            
            # Add wake word configuration if provided
            if self.wake_words:
                # Use OpenWakeWord backend which supports custom wake words
                wake_words_str = ", ".join(self.wake_words) if isinstance(self.wake_words, list) else self.wake_words
                config["wake_words"] = wake_words_str
                config["wakeword_backend"] = "oww"  # Use OpenWakeWord
                config["on_wakeword_detected"] = self._on_wake_word_detected
                config["openwakeword_inference_framework"] = "onnx"  # Use ONNX for better compatibility
                logger.info(f"Using OpenWakeWord backend with wake word: {wake_words_str}")
            
            self.recorder = AudioToTextRecorder(**config)
            logger.info(f"Voice controller initialized with model: {self.model_size}")
        except Exception as e:
            logger.error(f"Failed to initialize recorder: {e}")
            raise
    
    def _on_wake_word_detected(self):
        logger.info("Wake word detected!")
        if self.on_wake_word_callback:
            self.on_wake_word_callback()
    
    def _on_transcription_update(self, text: str):
        if text and len(text.strip()) > 0:
            logger.debug(f"Transcription update: {text}")
    
    def start(self):
        if self.is_listening:
            logger.warning("Already listening")
            return
            
        self.is_listening = True
        logger.info("Starting voice controller...")
        
        try:
            while self.is_listening:
                if self.wake_words:
                    logger.info("Listening for wake word...")
                else:
                    logger.info("Listening for command...")
                text = self.recorder.text()
                
                if text and len(text.strip()) > 0:
                    logger.info(f"Recognized: {text}")
                    if self.on_command_callback:
                        self.on_command_callback(text)
                    
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Stopping voice controller...")
            self.stop()
        except Exception as e:
            logger.error(f"Error in voice controller: {e}")
            self.stop()
            raise
    
    def stop(self):
        self.is_listening = False
        if self.recorder:
            try:
                self.recorder.stop()
            except:
                pass
        logger.info("Voice controller stopped")
    
    def start_async(self):
        thread = threading.Thread(target=self.start, daemon=True)
        thread.start()
        return thread
import os
import threading
import time
from typing import Optional, Callable, Dict, Any, List
from RealtimeSTT import AudioToTextRecorder
import logging
from . import voice_config
from .language_aware_recorder import LanguageAwareRecorder

logger = logging.getLogger(__name__)


class DuckVoiceController:
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        model_size: str = "tiny",
        language: str = "en",
        wake_words: list = None,
        on_wake_word: Optional[Callable] = None,
        on_command: Optional[Callable[[str], None]] = None,
        filter_languages: bool = None,
        allowed_languages: List[str] = None
    ):
        self.api_url = api_url
        self.model_size = model_size
        self.language = language
        self.wake_words = wake_words
        self.on_wake_word_callback = on_wake_word
        self.on_command_callback = on_command
        self.is_listening = False
        self.recorder = None
        
        # Language filtering configuration
        self.filter_languages = filter_languages if filter_languages is not None else voice_config.FILTER_NON_ENGLISH
        self.allowed_languages = allowed_languages if allowed_languages is not None else voice_config.ALLOWED_LANGUAGES
        self.language_detection_enabled = voice_config.LANGUAGE_DETECTION_ENABLED
        self.language_confidence_threshold = voice_config.LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD
        
        self._init_recorder()
        
    def _init_recorder(self):
        try:
            # Base configuration
            config = {
                "model": self.model_size,
                # Don't set language to allow automatic detection
                "language": "" if self.filter_languages else self.language,
                "silero_sensitivity": 0.4,
                "webrtc_sensitivity": 3,
                "post_speech_silence_duration": 0.4,
                "min_length_of_recording": 0.5,
                "min_gap_between_recordings": 0.3,
                "enable_realtime_transcription": True,
                "realtime_processing_pause": 0.2,
                "realtime_model_type": self.model_size,
                "on_realtime_transcription_update": self._on_transcription_update,
                "on_recorded_chunk": self._on_recorded_chunk if self.language_detection_enabled else None,
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
            
            if self.filter_languages:
                # Use our language-aware recorder
                self.recorder = LanguageAwareRecorder(
                    config=config,
                    allowed_languages=self.allowed_languages,
                    filter_languages=True
                )
            else:
                # Use standard recorder
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
    
    def _on_recorded_chunk(self, chunk):
        """Callback for when audio chunk is recorded, used for language detection"""
        # This is called before transcription, we'll use it to prepare for language filtering
        # The chunk parameter is not used in this simple implementation
        pass
    
    def _is_language_allowed(self, text: str, detected_language: Optional[str] = None) -> bool:
        """Check if the detected language is in the allowed list"""
        if not self.filter_languages:
            return True
        
        # If we have a detected language, use it
        if detected_language:
            is_allowed = detected_language in self.allowed_languages
            if not is_allowed:
                logger.info(f"Filtered out text in language '{detected_language}': {text}")
            return is_allowed
        
        # Simple heuristic: check if text contains mostly ASCII characters (for English)
        if "en" in self.allowed_languages:
            ascii_chars = sum(1 for c in text if ord(c) < 128)
            total_chars = len(text)
            if total_chars > 0:
                ascii_ratio = ascii_chars / total_chars
                is_english = ascii_ratio > 0.8  # 80% ASCII characters suggests English
                if not is_english:
                    logger.info(f"Filtered out non-English text (ASCII ratio: {ascii_ratio:.2f}): {text}")
                return is_english
        
        return True
    
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
                    # Get detected language info if available
                    detected_language = None
                    try:
                        # Try to access detected language from our custom recorder
                        if hasattr(self.recorder, 'detected_language'):
                            detected_language = self.recorder.detected_language
                            confidence = getattr(self.recorder, 'language_probability', 0.0)
                            logger.info(f"Recognized: {text} (language: {detected_language or 'unknown'}, confidence: {confidence:.2f})")
                        else:
                            logger.info(f"Recognized: {text}")
                    except:
                        logger.info(f"Recognized: {text}")
                    
                    # The language filtering is already done in LanguageAwareRecorder
                    # If we get text here, it means it passed the filter
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
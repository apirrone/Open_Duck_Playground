"""
Language-aware audio recorder that can detect and filter languages
"""
import logging
from typing import Optional, Callable, List
from RealtimeSTT import AudioToTextRecorder

logger = logging.getLogger(__name__)


class LanguageAwareRecorder:
    """Wrapper around AudioToTextRecorder that adds language detection and filtering"""
    
    def __init__(self, config: dict, allowed_languages: List[str] = None, filter_languages: bool = True):
        self.allowed_languages = allowed_languages or ["en"]
        self.filter_languages = filter_languages
        self.detected_language = None
        self.language_probability = 0.0
        self._last_text = ""
        self._original_callback = config.get("on_realtime_transcription_update")
        
        # Override the callback to intercept transcriptions
        config["on_realtime_transcription_update"] = self._intercept_transcription
        
        # Initialize the base recorder
        self.recorder = AudioToTextRecorder(**config)
        
    def _intercept_transcription(self, text: str):
        """Intercept transcription to detect language"""
        self._last_text = text
        if self._original_callback:
            self._original_callback(text)
    
    def text(self) -> str:
        """Get transcribed text with language detection"""
        # Get the transcription
        text = self.recorder.text()
        
        if not text or not self.filter_languages:
            return text
            
        # Try to detect language from the model's internal state
        try:
            # Access the model's detected language if available
            if hasattr(self.recorder, 'transcribe_worker') and hasattr(self.recorder.transcribe_worker, 'detected_language'):
                self.detected_language = self.recorder.transcribe_worker.detected_language
                self.language_probability = getattr(self.recorder.transcribe_worker, 'language_probability', 0.0)
                
                # Check if language is allowed
                if self.detected_language and self.detected_language not in self.allowed_languages:
                    logger.info(f"Filtered out {self.detected_language} text (confidence: {self.language_probability:.2f}): {text}")
                    return ""  # Return empty string to ignore this transcription
        except Exception as e:
            logger.debug(f"Could not detect language: {e}")
            
        return text
    
    def __getattr__(self, name):
        """Delegate all other methods to the base recorder"""
        return getattr(self.recorder, name)
import logging
import threading
import time
from typing import Optional, Callable, List, Union
import pvporcupine
from pvrecorder import PvRecorder

logger = logging.getLogger(__name__)


class PorcupineWakeWord:
    def __init__(
        self,
        access_key: str,
        keywords: Union[List[str], List[str]] = None,
        keyword_paths: List[str] = None,
        sensitivities: List[float] = None,
        on_wake_word: Optional[Callable[[int, str], None]] = None
    ):
        self.access_key = access_key
        self.keywords = keywords
        self.keyword_paths = keyword_paths
        self.sensitivities = sensitivities
        self.on_wake_word_callback = on_wake_word
        
        self.porcupine = None
        self.recorder = None
        self.is_listening = False
        self.thread = None
        
        self._init_porcupine()
        
    def _init_porcupine(self):
        try:
            if self.keywords:
                self.porcupine = pvporcupine.create(
                    access_key=self.access_key,
                    keywords=self.keywords,
                    sensitivities=self.sensitivities
                )
                logger.info(f"Initialized Porcupine with built-in keywords: {self.keywords}")
            elif self.keyword_paths:
                self.porcupine = pvporcupine.create(
                    access_key=self.access_key,
                    keyword_paths=self.keyword_paths,
                    sensitivities=self.sensitivities
                )
                logger.info(f"Initialized Porcupine with custom keyword models: {self.keyword_paths}")
            else:
                raise ValueError("Either keywords or keyword_paths must be provided")
                
            self.recorder = PvRecorder(
                device_index=-1,
                frame_length=self.porcupine.frame_length
            )
            
            logger.info(f"Porcupine initialized successfully with frame length: {self.porcupine.frame_length}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}")
            raise
    
    def start(self):
        if self.is_listening:
            logger.warning("Already listening for wake word")
            return
            
        self.is_listening = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        logger.info("Started listening for wake word")
        
    def stop(self):
        self.is_listening = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.recorder:
            try:
                self.recorder.stop()
                self.recorder.delete()
            except:
                pass
                
        if self.porcupine:
            self.porcupine.delete()
            
        logger.info("Stopped wake word detection")
        
    def _listen_loop(self):
        try:
            self.recorder.start()
            logger.info("Porcupine recorder started")
            
            while self.is_listening:
                pcm = self.recorder.read()
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    if self.keywords:
                        detected_word = self.keywords[keyword_index]
                    else:
                        detected_word = f"custom_keyword_{keyword_index}"
                    
                    logger.info(f"Wake word detected: {detected_word} (index: {keyword_index})")
                    
                    if self.on_wake_word_callback:
                        self.on_wake_word_callback(keyword_index, detected_word)
                        
        except Exception as e:
            logger.error(f"Error in wake word detection loop: {e}")
            self.is_listening = False
            
    def is_active(self) -> bool:
        return self.is_listening


class PorcupineWakeWordDetector:
    @staticmethod
    def get_available_keywords() -> List[str]:
        return [
            "alexa", "americano", "blueberry", "bumblebee", "computer",
            "grapefruit", "grasshopper", "hey google", "hey siri", 
            "jarvis", "ok google", "picovoice", "porcupine", "terminator"
        ]
    
    @staticmethod
    def create_for_keyword(
        access_key: str, 
        keyword: str,
        sensitivity: float = 0.5,
        on_wake_word: Optional[Callable[[str], None]] = None
    ) -> PorcupineWakeWord:
        def callback_wrapper(index: int, word: str):
            if on_wake_word:
                on_wake_word(word)
                
        return PorcupineWakeWord(
            access_key=access_key,
            keywords=[keyword],
            sensitivities=[sensitivity],
            on_wake_word=callback_wrapper
        )
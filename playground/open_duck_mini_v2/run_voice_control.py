#!/usr/bin/env python3
import argparse
import logging
import signal
import sys
import time
from playground.open_duck_mini_v2.voice_controller import DuckVoiceController
from playground.open_duck_mini_v2.voice_commands import DuckCommandParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global controller for signal handling
controller = None


def signal_handler(sig, frame):
    logger.info("\nShutting down voice control...")
    if controller:
        controller.stop()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Voice control for Open Duck robot")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API endpoint URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: small)"
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code (default: en)"
    )
    parser.add_argument(
        "--wake-word",
        default="duck duck",
        help="Wake word phrase (default: 'duck duck'). For Porcupine, use built-in keywords like 'computer', 'jarvis', 'alexa', etc."
    )
    parser.add_argument(
        "--no-wake-word",
        action="store_true",
        help="Run in always-listening mode without a wake word."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--auto-stop-delay",
        type=float,
        default=5.0,
        help="Seconds before a movement command is automatically stopped (default: 5.0)"
    )
    parser.add_argument(
        "--wake-word-backend",
        default="porcupine",
        choices=["oww", "porcupine"],
        help="Wake word detection backend (default: porcupine)"
    )
    parser.add_argument(
        "--porcupine-access-key",
        help="Picovoice access key for Porcupine wake word detection. Get one at https://console.picovoice.ai/"
    )
    parser.add_argument(
        "--porcupine-keyword-path",
        help="Path to custom Porcupine wake word model (.ppn file)"
    )
    parser.add_argument(
        "--wake-word-sensitivity",
        type=float,
        default=0.5,
        help="Wake word detection sensitivity (0.0-1.0, default: 0.5)"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger("RealtimeSTT").setLevel(logging.DEBUG)
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("Starting Open Duck Voice Control")
    logger.info(f"API URL: {args.api_url}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Language: {args.language}")
    
    use_wake_word = not args.no_wake_word
    if use_wake_word:
        logger.info(f"Wake word: {args.wake_word}")
        logger.info(f"Wake word backend: {args.wake_word_backend}")
    else:
        logger.info("Wake word: None (always listening)")
        
    # Validate Porcupine settings
    if use_wake_word and args.wake_word_backend == "porcupine" and not args.porcupine_access_key:
        print("\n❌ Error: Porcupine access key required when using Porcupine backend")
        print("Get your free access key at: https://console.picovoice.ai/")
        print("Then run with: --porcupine-access-key YOUR_ACCESS_KEY")
        sys.exit(1)

    # Initialize command parser
    command_parser = DuckCommandParser(
        api_url=args.api_url,
        auto_stop_delay=args.auto_stop_delay
    )
    
    # Initialize voice controller
    global controller
    
    # Prepare keyword paths for Porcupine if provided
    porcupine_keyword_paths = None
    if args.porcupine_keyword_path:
        porcupine_keyword_paths = [args.porcupine_keyword_path]
    
    controller = DuckVoiceController(
        api_url=args.api_url,
        model_size=args.model,
        language=args.language,
        wake_words=[args.wake_word] if use_wake_word else None,
        wake_word_backend=args.wake_word_backend if use_wake_word else None,
        porcupine_access_key=args.porcupine_access_key,
        porcupine_keyword_paths=porcupine_keyword_paths,
        wake_word_sensitivity=args.wake_word_sensitivity,
        on_wake_word=lambda: print("\n🦆 Wake word detected! Listening for command..."),
        on_command=lambda text: handle_command(text, command_parser)
    )
    
    print("\n" + "="*50)
    print("🦆 Open Duck Voice Control Started")
    print("="*50)
    
    if not use_wake_word:
        print("Always listening mode - speak your commands directly:")
        print("\nExample commands:")
        print("  - 'go forward'")
        print("  - 'turn left slowly'")
        print("  - 'look up'")
        print("  - 'stop'")
        print("  - 'switch to head mode'")
    else:
        print(f"Say '{args.wake_word}' followed by a command:")
        print("\nExample commands:")
        print(f"  - '{args.wake_word}, go forward'")
        print(f"  - '{args.wake_word}, turn left slowly'")
        print(f"  - '{args.wake_word}, look up'")
        print(f"  - '{args.wake_word}, stop'")
        print(f"  - '{args.wake_word}, switch to head mode'")
    
    print("\nPress Ctrl+C to exit")
    print("="*50 + "\n")
    
    try:
        controller.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


def handle_command(text: str, parser: DuckCommandParser):
    print(f"\n🎤 Heard: '{text}'")
    
    result = parser.execute_command(text)
    
    if result["success"]:
        print(f"✅ {result['message']}")
    else:
        print(f"❌ {result['message']}")
    
    print("\n🦆 Ready for next command...")


if __name__ == "__main__":
    main()
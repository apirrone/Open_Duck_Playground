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
        "--filter-languages",
        action="store_true",
        help="Enable language filtering to only accept specified languages"
    )
    parser.add_argument(
        "--allowed-languages",
        nargs="+",
        default=["en"],
        help="List of allowed language codes when filtering is enabled (default: en)"
    )
    parser.add_argument(
        "--wake-word",
        default="duck duck",
        help="Wake word phrase (default: 'duck duck'). OpenWakeWord supports custom phrases."
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
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger("RealtimeSTT").setLevel(logging.DEBUG)
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("Starting Open Duck Voice Control")
    logger.info(f"API URL: {args.api_url}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Language: {args.language}")
    if args.filter_languages:
        logger.info(f"Language filtering enabled. Allowed languages: {', '.join(args.allowed_languages)}")

    use_wake_word = not args.no_wake_word
    if use_wake_word:
        logger.info(f"Wake word: {args.wake_word}")
    else:
        logger.info("Wake word: None (always listening)")

    # Initialize command parser
    command_parser = DuckCommandParser(api_url=args.api_url)
    
    # Initialize voice controller
    global controller
    controller = DuckVoiceController(
        api_url=args.api_url,
        model_size=args.model,
        language=args.language,
        wake_words=[args.wake_word] if use_wake_word else None,
        on_wake_word=lambda: print("\n🦆 Wake word detected! Listening for command..."),
        on_command=lambda text: handle_command(text, command_parser),
        filter_languages=args.filter_languages,
        allowed_languages=args.allowed_languages
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
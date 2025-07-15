#!/usr/bin/env python3
import argparse
import time
import signal
import sys
from playground.open_duck_mini_v2.porcupine_wake_word import PorcupineWakeWordDetector, PorcupineWakeWord


def signal_handler(sig, frame):
    print("\nExiting...")
    sys.exit(0)


def test_wake_word_detection(access_key: str, keyword: str = "computer", sensitivity: float = 0.5):
    print(f"\n{'='*60}")
    print("🎤 Porcupine Wake Word Detection Test")
    print(f"{'='*60}")
    
    # List available keywords
    available_keywords = PorcupineWakeWordDetector.get_available_keywords()
    print(f"\nAvailable built-in keywords: {', '.join(available_keywords)}")
    
    # Check if keyword is available
    if keyword not in available_keywords:
        print(f"\n⚠️  Warning: '{keyword}' is not a built-in keyword.")
        print("You may need to provide a custom .ppn model file.")
        return
    
    print(f"\nTesting with keyword: '{keyword}'")
    print(f"Sensitivity: {sensitivity}")
    print("\nSpeak the wake word to test detection...")
    print("Press Ctrl+C to exit\n")
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create detector
    detector = None
    detection_count = 0
    
    def on_wake_word_detected(word: str):
        nonlocal detection_count
        detection_count += 1
        print(f"✅ Wake word detected: '{word}' (Detection #{detection_count})")
        print("   Listening again...\n")
    
    try:
        detector = PorcupineWakeWordDetector.create_for_keyword(
            access_key=access_key,
            keyword=keyword,
            sensitivity=sensitivity,
            on_wake_word=on_wake_word_detected
        )
        
        # Start detection
        detector.start()
        
        # Keep running
        while True:
            time.sleep(0.1)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have a valid Picovoice access key")
        print("2. Get one free at: https://console.picovoice.ai/")
        print("3. Check your internet connection (for first-time model download)")
        
    finally:
        if detector:
            detector.stop()
            print("\n\nDetector stopped.")


def test_custom_model(access_key: str, model_path: str, sensitivity: float = 0.5):
    print(f"\n{'='*60}")
    print("🎤 Porcupine Custom Wake Word Test")
    print(f"{'='*60}")
    
    print(f"\nTesting with custom model: {model_path}")
    print(f"Sensitivity: {sensitivity}")
    print("\nSpeak your custom wake word to test detection...")
    print("Press Ctrl+C to exit\n")
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create detector
    detector = None
    detection_count = 0
    
    def on_wake_word_detected(idx: int, word: str):
        nonlocal detection_count
        detection_count += 1
        print(f"✅ Custom wake word detected! (Detection #{detection_count})")
        print("   Listening again...\n")
    
    try:
        detector = PorcupineWakeWord(
            access_key=access_key,
            keyword_paths=[model_path],
            sensitivities=[sensitivity],
            on_wake_word=on_wake_word_detected
        )
        
        # Start detection
        detector.start()
        
        # Keep running
        while True:
            time.sleep(0.1)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the .ppn file path is correct")
        print("2. Ensure the model was created for your platform")
        print("3. Check that your access key is valid")
        
    finally:
        if detector:
            detector.stop()
            print("\n\nDetector stopped.")


def main():
    parser = argparse.ArgumentParser(description="Test Porcupine wake word detection")
    parser.add_argument(
        "--access-key",
        required=True,
        help="Picovoice access key (get one at https://console.picovoice.ai/)"
    )
    parser.add_argument(
        "--keyword",
        default="computer",
        help="Built-in keyword to test (default: computer)"
    )
    parser.add_argument(
        "--custom-model",
        help="Path to custom wake word model (.ppn file)"
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.5,
        help="Detection sensitivity (0.0-1.0, default: 0.5)"
    )
    parser.add_argument(
        "--list-keywords",
        action="store_true",
        help="List all available built-in keywords and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_keywords:
        keywords = PorcupineWakeWordDetector.get_available_keywords()
        print("\nAvailable built-in Porcupine keywords:")
        for kw in keywords:
            print(f"  - {kw}")
        print("\nUse any of these with --keyword")
        return
    
    if args.custom_model:
        test_custom_model(args.access_key, args.custom_model, args.sensitivity)
    else:
        test_wake_word_detection(args.access_key, args.keyword, args.sensitivity)


if __name__ == "__main__":
    main()
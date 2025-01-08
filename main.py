#!/usr/bin/env python3

import sys
import queue
import threading
import time

# --- Keyboard-related imports ---
from evdev import InputDevice, categorize, ecodes, list_devices
from pynput.keyboard import Controller as KeyboardController, Key

# --- Audio/transcription-related imports ---
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

################################################################################
# 1) Keyboard Device Detection
################################################################################

def find_keyboard_device():
    """
    Tries to find an event device that looks like a keyboard.
    In practice, you might have to hardcode the event number
    or select from a list if multiple keyboards exist.
    """
    from evdev import list_devices
    for dev_path in list_devices():
        device = InputDevice(dev_path)
        capabilities = device.capabilities(verbose=True)
        # We're looking for a device that has EV_KEY in its capabilities
        # and name = 'USB-HID Keyboard' (adjust if your device name differs).
        if ('EV_KEY', 1) in capabilities and device.name == 'USB-HID Keyboard':
            return dev_path
    return None

################################################################################
# 2) Global variables and flags
################################################################################

is_recording = False         # Indicates if we're currently recording
stop_recording_flag = False  # Signals the recording thread to stop
audio_queue = queue.Queue()  # Thread-safe queue to store recorded chunks
record_thread = None         # Reference to the recording thread

# Track Ctrl keys for normal start/stop toggling
left_ctrl_pressed = False
right_ctrl_pressed = False

# Track Shift keys for aborting ongoing recording
left_shift_pressed = False
right_shift_pressed = False

# We'll create a single Whisper model once, so we don't re-load it every time.
#  tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, large-v3, large, distil-large-v2, distil-medium.en, distil-small.en, distil-large-v3, large-v3-turbo, turbo
whisper_model = WhisperModel("large-v3-turbo", device="cuda", compute_type="default")

# For simulating keystrokes
keyboard_simulator = KeyboardController()

################################################################################
# 3) Audio Recording Thread
################################################################################

def audio_recording_thread(sample_rate=16000):
    """
    Runs in a separate thread; reads the microphone input and stores data in 'audio_queue'.
    Stops when 'stop_recording_flag' is set to True.
    """
    global stop_recording_flag

    def callback(indata, frames, time_info, status):
        if status:
            print(f"SoundDevice status: {status}", file=sys.stderr)
        audio_queue.put(indata.copy())  # Add a copy of the audio chunk to the queue

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16',
                        callback=callback, blocksize=1024):
        print("[Recording thread] Started recording.")
        while True:
            if stop_recording_flag:
                print("[Recording thread] Stop flag received.")
                break
            time.sleep(0.01)

    print("[Recording thread] Exiting.")

################################################################################
# 4) Start/Stop Recording Helpers
################################################################################

def start_recording():
    """Initialize audio queue and start the recording thread."""
    global is_recording, stop_recording_flag, record_thread, audio_queue

    if is_recording:
        print("[Main] Already recording.")
        return

    print("[Main] Starting recording...")
    is_recording = True
    stop_recording_flag = False
    audio_queue = queue.Queue()
    record_thread = threading.Thread(target=audio_recording_thread, args=(16000,))
    record_thread.daemon = True
    record_thread.start()

def stop_recording_and_transcribe():
    """
    Signal the recording thread to stop, collect audio, run transcription,
    then type out the result.
    """
    global is_recording, stop_recording_flag, record_thread

    if not is_recording:
        print("[Main] Not currently recording.")
        return

    print("[Main] Stopping recording...")
    is_recording = False
    stop_recording_flag = True
    if record_thread is not None:
        record_thread.join()

    # Collect all audio from the queue
    print("[Main] Collecting audio data...")
    recorded_chunks = []
    while not audio_queue.empty():
        recorded_chunks.append(audio_queue.get())

    if not recorded_chunks:
        print("[Main] No audio recorded.")
        return

    # Concatenate all chunks into one numpy array
    audio_data = np.concatenate(recorded_chunks, axis=0).flatten()

    # Transcribe
    print("[Main] Transcribing audio...")
    text_result = transcribe_audio(audio_data, sample_rate=16000)
    print(f"[Main] Transcription result: {text_result!r}")

    # Type the result
    simulate_typing(text_result)

def abort_recording():
    """
    Abort any ongoing recording without transcription.
    This stops the recording thread and discards the recorded audio.
    """
    global is_recording, stop_recording_flag, record_thread, audio_queue

    if not is_recording:
        # Nothing to abort if we're not currently recording
        return

    print("[Main] Aborting recording (no transcription).")
    is_recording = False
    stop_recording_flag = True

    if record_thread is not None:
        record_thread.join()

    # Discard any recorded audio
    print("[Main] Discarding recorded audio data...")
    while not audio_queue.empty():
        audio_queue.get()

def transcribe_audio(audio_data, sample_rate=16000):
    """
    Transcribe an int16 numpy array using faster_whisper.
    Returns the resulting text string.
    """
    # Convert to float32 for the model
    audio_data_float32 = audio_data.astype(np.float32) / 32768.0
    segments, _info = whisper_model.transcribe(
        audio=audio_data_float32,
        language="en",
        word_timestamps=False,
        beam_size=5,
    )

    # Combine all segments text
    transcription = "".join(segment.text for segment in segments).strip()
    return transcription

################################################################################
# 5) Typing with Capitalization Preserved
################################################################################

def simulate_typing(text):
    """
    Simulate typing of the transcribed text using pynput,
    preserving capitalization by pressing Shift for uppercase letters
    and for punctuation that requires Shift.
    """
    SHIFT_REQUIRED = {
        '?': '/',
        ':': ';',
        '"': "'",
        '<': ',',
        '>': '.',
        '{': '[',
        '}': ']',
        '|': '\\',
        '~': '`',
        '!': '1',
        '@': '2',
        '#': '3',
        '$': '4',
        '%': '5',
        '^': '6',
        '&': '7',
        '*': '8',
        '(': '9',
        ')': '0',
        '_': '-',
        '+': '='
    }

    for ch in text:
        # 1) Handle punctuation/symbols that need Shift
        if ch in SHIFT_REQUIRED:
            with keyboard_simulator.pressed(Key.shift):
                key_to_press = SHIFT_REQUIRED[ch]
                keyboard_simulator.press(key_to_press)
                keyboard_simulator.release(key_to_press)

        # 2) Handle uppercase letters
        elif ch.isalpha() and ch.isupper():
            with keyboard_simulator.pressed(Key.shift):
                keyboard_simulator.press(ch.lower())
                keyboard_simulator.release(ch.lower())

        # 3) Type everything else “as is”
        else:
            keyboard_simulator.press(ch)
            keyboard_simulator.release(ch)
        
        # Optional short delay:
        # time.sleep(0.01)

################################################################################
# 6) Keyboard Event Loop
################################################################################

def keyboard_listener_loop(dev_path):
    """
    Reads events from the keyboard device and:
      - Toggles start/stop recording when Left Ctrl + Right Ctrl are pressed.
      - Aborts any ongoing recording when Left Shift + Right Shift are pressed.
    """
    global left_ctrl_pressed, right_ctrl_pressed
    global left_shift_pressed, right_shift_pressed

    dev = InputDevice(dev_path)
    print(f"[Main] Listening on keyboard device: {dev_path}")

    print("[Main] Press Left Ctrl + Right Ctrl together to start/stop recording.")
    print("[Main] Press Left Shift + Right Shift together to ABORT any recording.")

    try:
        for event in dev.read_loop():
            if event.type == ecodes.EV_KEY:
                key_event = categorize(event)

                # Handle Ctrl keys
                if key_event.keycode == "KEY_LEFTCTRL":
                    if key_event.keystate == key_event.key_down:
                        left_ctrl_pressed = True
                    elif key_event.keystate == key_event.key_up:
                        left_ctrl_pressed = False

                if key_event.keycode == "KEY_RIGHTCTRL":
                    if key_event.keystate == key_event.key_down:
                        right_ctrl_pressed = True
                    elif key_event.keystate == key_event.key_up:
                        right_ctrl_pressed = False

                # Handle Shift keys
                if key_event.keycode == "KEY_LEFTSHIFT":
                    if key_event.keystate == key_event.key_down:
                        left_shift_pressed = True
                    elif key_event.keystate == key_event.key_up:
                        left_shift_pressed = False

                if key_event.keycode == "KEY_RIGHTSHIFT":
                    if key_event.keystate == key_event.key_down:
                        right_shift_pressed = True
                    elif key_event.keystate == key_event.key_up:
                        right_shift_pressed = False

                # Check for simultaneous Ctrl presses -> toggle recording
                if left_ctrl_pressed and right_ctrl_pressed and key_event.keystate == key_event.key_down:
                    if not is_recording:
                        start_recording()
                    else:
                        stop_recording_and_transcribe()

                # Check for simultaneous Shift presses -> abort recording
                if left_shift_pressed and right_shift_pressed and key_event.keystate == key_event.key_down:
                    abort_recording()

    except KeyboardInterrupt:
        print("[Main] KeyboardInterrupt received. Exiting.")
    finally:
        # If still recording upon exit, stop gracefully (with transcription or abort).
        if is_recording:
            stop_recording_and_transcribe()
        print("[Main] Bye!")

################################################################################
# 7) Main Entry Point
################################################################################

def main():
    dev_path = find_keyboard_device()  # adjust if your device name is different
    if not dev_path:
        print("Could not find a keyboard-like device. Exiting.", file=sys.stderr)
        sys.exit(1)

    keyboard_listener_loop(dev_path)

if __name__ == "__main__":
    main()

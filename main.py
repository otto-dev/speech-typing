#!/usr/bin/env python3

import sys
import queue
import threading
import time
import subprocess
import tkinter as tk

# --- Keyboard-related imports ---
from evdev import InputDevice, categorize, ecodes, list_devices
from pynput.keyboard import Controller as KeyboardController, Key

# --- Audio/transcription-related imports ---
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

################################################################################
# 0) Activity Indicator Overlay using Tkinter
################################################################################

def get_primary_display_geometry():
    """
    Use xrandr to find the geometry (x, y, width, height) of the primary display.
    Returns (disp_x, disp_y, disp_width, disp_height).

    If it can't detect the primary display for any reason, returns None.
    """
    try:
        result = subprocess.run(
            ["xrandr", "--query"], capture_output=True, text=True, check=True
        ).stdout

        # Look for a line containing " primary "
        for line in result.splitlines():
            if " primary " in line:
                # Typical line example: "DP-1 connected primary 1920x1080+0+0 ..."
                parts = line.split()
                # Find the part that looks like 1920x1080+0+0
                for part in parts:
                    if "+" in part and "x" in part:
                        geometry = part
                        break
                else:
                    continue

                # Parse the geometry "1920x1080+0+0"
                res, x_off, y_off = geometry.split("+")
                width, height = res.split("x")
                disp_width = int(width)
                disp_height = int(height)
                disp_x = int(x_off)
                disp_y = int(y_off)

                return (disp_x, disp_y, disp_width, disp_height)
        return None
    except Exception:
        return None

# We'll create a global Tk root for our overlay:
root = None
listening_label = None

def init_overlay():
    """
    Initializes the Tkinter overlay (but does not show it by default).
    """
    global root, listening_label

    root = tk.Tk()
    root.overrideredirect(True)       # Remove window decorations
    root.attributes("-topmost", True) # Keep on top


    # Dimensions of our overlay
    overlay_width = 150
    overlay_height = 50

    # Margin from the bottom and left edges
    margin = 40

    # Detect the geometry of the primary display, if available
    primary_geometry = get_primary_display_geometry()
    if primary_geometry is not None:
        disp_x, disp_y, disp_width, disp_height = primary_geometry
        # Bottom-left corner, with a margin
        x_pos = disp_x + margin
        y_pos = disp_y + disp_height - overlay_height - margin
    else:
        # Fallback: use total screen width/height from Tkinter
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x_pos = margin
        y_pos = screen_height - overlay_height - margin

    # Set geometry
    root.geometry(f"{overlay_width}x{overlay_height}+{x_pos}+{y_pos}")

    # Create the label
    listening_label = tk.Label(
        root,
        text="Listening...",
        fg="white",
        bg="green",
        font=("Helvetica", 12, "bold")
    )
    listening_label.pack(expand=True, fill="both")

    # Hide by default
    root.withdraw()

def show_overlay():
    """
    Makes the overlay visible.
    """
    if root is not None:
        root.deiconify()

def hide_overlay():
    """
    Hides the overlay.
    """
    if root is not None:
        root.withdraw()

################################################################################
# 1) Keyboard Device Detection
################################################################################

def find_keyboard_device():
    """
    Tries to find an event device that looks like a keyboard.
    In practice, you might have to hardcode the event number
    or select from a list if multiple keyboards exist.
    """
    for dev_path in list_devices():
        device = InputDevice(dev_path)
        capabilities = device.capabilities(verbose=True)
        # Adjust the device.name check if your keyboard has a different name.
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

    # Show "listening..." overlay
    show_overlay()

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

    # Hide "listening..." overlay
    hide_overlay()

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

    # Hide "listening..." overlay
    hide_overlay()

    # Discard any recorded audio
    print("[Main] Discarding recorded audio data...")
    while not audio_queue.empty():
        audio_queue.get()

def transcribe_audio(audio_data, sample_rate=16000):
    """
    Transcribe an int16 numpy array using faster_whisper.
    Returns the resulting text string.
    """
    audio_data_float32 = audio_data.astype(np.float32) / 32768.0
    segments, _info = whisper_model.transcribe(
        audio=audio_data_float32,
        language="en",
        word_timestamps=False,
        beam_size=5,
    )
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

        # 3) Type everything else as-is
        else:
            keyboard_simulator.press(ch)
            keyboard_simulator.release(ch)

################################################################################
# 6) Keyboard Event Loop
################################################################################

def keyboard_listener_loop(dev_path):
    """
    Reads events from the keyboard device and:
      - Toggles start/stop recording when Left Ctrl + Right Ctrl are pressed.
      - Aborts any ongoing recording when Left Shift + Right Shift are pressed.
      - If OSError: [Errno 19] No such device, go into inactive mode and wait for
        a new keyboard. Once found, automatically resume listening on that new device.
    """
    global left_ctrl_pressed, right_ctrl_pressed
    global left_shift_pressed, right_shift_pressed

    print(f"[Main] Listening on keyboard device: {dev_path}")
    dev = InputDevice(dev_path)

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

    except OSError as e:
        # Gracefully handle "No such device"
        if e.errno == 19:
            print("[Main] Device not found or disconnected. Inactive mode.")
            print("[Main] Will automatically resume when a new keyboard is found.")
            # Inactive mode: poll for a newly detected keyboard
            while True:
                time.sleep(1)
                possible_new_device = find_keyboard_device()
                if possible_new_device:
                    print("[Main] Found a new keyboard. Resuming operation.")
                    keyboard_listener_loop(possible_new_device)
                    return  # Important to exit the old function scope
        else:
            # If it's some other OSError, re-raise
            raise

    except KeyboardInterrupt:
        print("[Main] KeyboardInterrupt received. Exiting.")
    finally:
        # If still recording upon exit, stop gracefully (with transcription).
        if is_recording:
            stop_recording_and_transcribe()
        print("[Main] Bye!")

################################################################################
# 7) Main Entry Point
################################################################################

def start_keyboard_thread():
    """
    Finds a keyboard device (or waits for one), then starts the keyboard listener loop.
    This runs in a separate thread so the Tk mainloop remains free.
    """
    def _run():
        dev_path = find_keyboard_device()
        if not dev_path:
            print("Could not find a keyboard-like device initially. Inactive mode.")
            print("Reconnect your keyboard at any time; script will automatically resume.")
            # Poll for a keyboard device
            while True:
                time.sleep(1)
                possible_new_device = find_keyboard_device()
                if possible_new_device:
                    print("[Main] Found a device. Now listening.")
                    keyboard_listener_loop(possible_new_device)
                    break
        else:
            keyboard_listener_loop(dev_path)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

def main():
    # 1) Initialize the Tkinter overlay
    init_overlay()

    # 2) Start keyboard listener in a background thread
    start_keyboard_thread()

    # 3) Run the Tkinter main loop in the main thread (blocking)
    print("[Main] Entering Tk mainloop. Press Ctrl+C in the terminal to quit.")
    root.mainloop()

if __name__ == "__main__":
    main()

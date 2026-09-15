"""
mind_controlled_sphero.py
─────────────────────────
Combines EEG/EMG wave classification with Sphero BOLT control.

Controls
────────
  1 JAW CLENCH    Robot rotates 45° (cumulative)
  2 JAW CLENCH    Robot starts/stops moving
"""

import time
import queue
import threading
import json

import numpy as np
import mss
import keyboard
from sklearn.ensemble import RandomForestClassifier

from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI
from spherov2.types import Color

# ──────────────────────────────────────────────
#  ML / Screen-capture config
# ──────────────────────────────────────────────
MONITOR = {"top": 180, "left": 1350, "width": 250, "height": 60}
BASELINE_Y = 30
COOLDOWN_FRAMES = 30      # frames to ignore after a detection (prevents double-firing)
AMPLITUDE_THRESHOLD = 40  # minimum amplitude to attempt classification
DOUBLE_CLENCH_WINDOW = 1.3  # seconds within which a second clench counts as a double

# ──────────────────────────────────────────────
#  Sphero drive config
# ──────────────────────────────────────────────
DRIVE_SPEED = 120          # constant forward speed while moving (0–255)
ROTATE_STEP = 45           # degrees per jaw clench
ROLL_DURATION = 0.03
CONTROL_INTERVAL = 0.02
SPHERO_NAME = "SB-E1E1"   # ← change to match your BOLT


# ══════════════════════════════════════════════
#  ML helpers  (unchanged from classifier.py)
# ══════════════════════════════════════════════

def extract_features(wave_array):
    data = np.array(wave_array)
    amplitude = np.max(data) - np.min(data)
    variance = np.var(data)
    jaggedness = np.sum(np.abs(np.diff(data)))
    return [amplitude, variance, jaggedness]


def get_wave_snapshot(img_array):
    wave = []
    for x in range(img_array.shape[1]):
        column = img_array[:, x, 0]
        max_brightness = np.max(column)
        if max_brightness > 50:
            wave.append(int(np.argmax(column)))
        else:
            wave.append(wave[-1] if wave else BASELINE_Y)
    return wave


def train_model():
    print("[ML] Loading training data from training_data.json …")
    try:
        with open("training_data.json", "r") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Could not find 'training_data.json'. "
            "Run the data-collection script first."
        )

    X, y = [], []
    for label, waves in dataset.items():
        for wave in waves:
            X.append(extract_features(wave))
            y.append(label)

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    print(f"[ML] Model trained on {len(X)} samples across {len(dataset)} classes.\n")
    return clf


# ══════════════════════════════════════════════
#  Classifier thread
#  Single clench → ROTATE
#  Double clench  (within DOUBLE_CLENCH_WINDOW) → MOVE toggle
#  Pushes "ROTATE" or "MOVE" strings into cmd_queue
# ══════════════════════════════════════════════

def classifier_thread(clf, cmd_queue: queue.Queue, stop_event: threading.Event):
    sct = mss.mss()
    cooldown = 0

    last_clench_time = None   # timestamp of the most recent confirmed clench
    pending_rotate = False    # True while we're waiting to see if a 2nd clench arrives

    print("[ML] Classifier running. Watching the screen …")
    while not stop_event.is_set():
        now = time.monotonic()

        # ── Flush a pending single-clench once the window expires ──────────
        if pending_rotate and (now - last_clench_time) > DOUBLE_CLENCH_WINDOW:
            print("[ML] 🦷 Single clench confirmed → ROTATE")
            cmd_queue.put("ROTATE")
            pending_rotate = False

        img = np.array(sct.grab(MONITOR))
        current_wave = get_wave_snapshot(img)

        if cooldown > 0:
            cooldown -= 1
        else:
            features = extract_features(current_wave)

            if features[0] > AMPLITUDE_THRESHOLD:
                prediction = clf.predict([features])[0]

                if prediction == "ROTATE":
                    if pending_rotate:
                        # ── Second clench inside the window → MOVE toggle ──
                        elapsed = now - last_clench_time
                        print(f"[ML] 🦷🦷 Double clench ({elapsed:.2f}s) → MOVE toggle")
                        cmd_queue.put("MOVE")
                        pending_rotate = False
                    else:
                        # ── First clench – start waiting ───────────────────
                        print("[ML] 🦷 Clench detected – waiting for second …")
                        last_clench_time = now
                        pending_rotate = True

                    cooldown = COOLDOWN_FRAMES

                 #elif prediction == "MOVE":
                     #print("[ML] 🤨 EYEBROW RAISE → MOVE toggle")
                    #cmd_queue.put("MOVE")
                    #cooldown = COOLDOWN_FRAMES

        time.sleep(1 / 60)


# ══════════════════════════════════════════════
#  Sphero helpers
# ══════════════════════════════════════════════

def set_screen_color(droid, toy, color: Color) -> bool:
    """Try every known API path to set the LED matrix colour."""
    for attempt in (
        lambda: toy.set_compressed_frame_player_one_color(color.r, color.g, color.b),
        lambda: (
            _register_solid(droid, color),
            droid.play_matrix_animation(0, loop=True),
        ),
        lambda: droid.set_matrix_fill(0, 0, 7, 7, color),
        lambda: [droid.set_matrix_pixel(x, y, color) for x in range(8) for y in range(8)],
        lambda: droid.set_matrix_character("#", color),
    ):
        try:
            attempt()
            return True
        except Exception:
            pass
    return False


def _register_solid(droid, color: Color):
    frame = [[15] * 8 for _ in range(8)]
    palette = [Color(0, 0, 0)] + [color] * 15
    droid.register_matrix_animation([frame], palette, fps=12, transition=False)


# ══════════════════════════════════════════════
#  Main – Sphero control loop
# ══════════════════════════════════════════════

def main():
    # 1. Train the model first so we fail fast if data is missing.
    clf = train_model()

    # 2. Connect to the Sphero.
    print(f"[Sphero] Searching for '{SPHERO_NAME}' …")
    toy = scanner.find_toy(toy_name=SPHERO_NAME, timeout=20)
    if toy is None:
        raise RuntimeError(
            "Sphero not found. Close the Sphero Edu app, "
            "ensure Bluetooth is on, and wake the BOLT."
        )
    print("[Sphero] Connected!\n")

    # 3. Shared state between the classifier thread and the control loop.
    cmd_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    # 4. Start the classifier in a background thread.
    ml_thread = threading.Thread(
        target=classifier_thread,
        args=(clf, cmd_queue, stop_event),
        daemon=True,
    )
    ml_thread.start()

    # 5. Sphero control loop (runs on the main thread – safer for BLE).
    with SpheroEduAPI(toy) as droid:
        heading = 0          # current facing direction (0–359°)
        is_moving = False    # toggled by MOVE commands

        # Start with a green LED so it's obvious we're live.
        droid.set_main_led(Color(0, 255, 0))
        set_screen_color(droid, toy, Color(0, 255, 0))

        print("─" * 50)
        print("Ready!  JAW CLENCH → rotate 45°")
        print("        EYEBROW RAISE → start / stop")
        print("        ESC → quit")
        print("─" * 50)

        try:
            while not keyboard.is_pressed("esc"):
                # ── Process any pending ML commands ──────────────
                try:
                    while True:                        # drain the whole queue each tick
                        cmd = cmd_queue.get_nowait()

                        if cmd == "ROTATE":
                            heading = (heading + ROTATE_STEP) % 360
                            print(f"[Robot] Rotating → new heading {heading}°")

                            # Execute the rotation:
                            # Stop first so the heading change is clean,
                            # then re-roll if we were already moving.
                            droid.stop_roll()
                            droid.set_heading(heading)
                            time.sleep(0.1)            # brief pause to let hardware settle

                            # Flash blue to give physical feedback
                            droid.set_main_led(Color(0, 100, 255))
                            set_screen_color(droid, toy, Color(0, 100, 255))
                            time.sleep(0.15)
                            led = Color(255, 100, 0) if is_moving else Color(0, 255, 0)
                            droid.set_main_led(led)
                            set_screen_color(droid, toy, led)

                        elif cmd == "MOVE":
                            is_moving = not is_moving

                            if is_moving:
                                print("[Robot] ▶ Moving forward")
                                droid.set_main_led(Color(255, 100, 0))
                                set_screen_color(droid, toy, Color(255, 100, 0))
                            else:
                                print("[Robot] ■ Stopped")
                                droid.stop_roll()
                                droid.set_main_led(Color(0, 255, 0))
                                set_screen_color(droid, toy, Color(0, 255, 0))

                except queue.Empty:
                    pass   # no new commands this tick – that's fine

                # ── Drive the robot if it should be moving ───────
                if is_moving:
                    droid.roll(heading, DRIVE_SPEED, ROLL_DURATION)

                time.sleep(CONTROL_INTERVAL)

        except KeyboardInterrupt:
            pass

        finally:
            print("\n[Sphero] Shutting down …")
            stop_event.set()
            droid.stop_roll()
            droid.set_main_led(Color(255, 0, 0))
            set_screen_color(droid, toy, Color(255, 0, 0))

    print("Done.")


if __name__ == "__main__":
    main()
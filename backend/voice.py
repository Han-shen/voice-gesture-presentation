from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

import pyautogui
import speech_recognition as sr


def listen_commands(
    stop_event: Any,
    pause_event: Any,
    state: Dict[str, Any],
    state_lock: Any,
    log_fn: Callable[..., None],
    config: Dict[str, Any],
) -> None:
    """
    Voice recognition loop.
    Designed to be resilient to start/stop and avoid microphone assertion crashes.
    """

    energy_threshold = int(config.get("energy_threshold", 300))
    pause_threshold = float(config.get("pause_threshold", 0.6))
    phrase_time_limit = float(config.get("phrase_time_limit", 2.5))
    commands: Dict[str, str] = config.get("commands", {}) or {}
    voice_cooldown = float(config.get("cooldown", 1.0))
    listen_timeout = float(config.get("listen_timeout", 0.6))

    r = sr.Recognizer()
    r.energy_threshold = energy_threshold
    r.pause_threshold = pause_threshold
    r.dynamic_energy_threshold = False

    mic = None
    try:
        mic = sr.Microphone()
    except Exception as e:
        log_fn(f"Microphone init failed: {e}", kind="voice")
        with state_lock:
            state["voice"]["running"] = False
        return

    last_action_ts = 0.0
    last_listen_log_ts = 0.0
    last_command_key: Optional[str] = None

    # Keep the microphone context open for the lifetime of this thread,
    # so stop() is just event signaling (no cross-thread stream closing).
    try:
        with mic as source:
            # One-time noise calibration to speed up steady-state listening.
            try:
                r.adjust_for_ambient_noise(source, duration=0.25)
            except Exception:
                pass

            log_fn("Voice module ready", kind="voice")

            while not stop_event.is_set():
                if pause_event.is_set():
                    with state_lock:
                        state["voice"]["paused"] = True
                    time.sleep(0.05)
                    continue

                with state_lock:
                    state["voice"]["paused"] = False

                now = time.time()
                if now - last_listen_log_ts > 2.0:
                    log_fn("Listening...", kind="voice")
                    last_listen_log_ts = now

                try:
                    audio = r.listen(
                        source,
                        timeout=listen_timeout,
                        phrase_time_limit=phrase_time_limit,
                    )
                except sr.WaitTimeoutError:
                    continue
                except AssertionError as e:
                    # Common when audio backends are interrupted; keep running.
                    log_fn(f"Microphone assertion: {e}", kind="voice")
                    time.sleep(0.5)
                    continue
                except Exception:
                    continue

                try:
                    command_text = r.recognize_google(audio).lower()
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    log_fn(f"Speech recognition error: {e}", kind="voice")
                    time.sleep(0.25)
                    continue
                except Exception:
                    continue

                # Match configured phrases.
                matched_phrase: Optional[str] = None
                matched_key: Optional[str] = None
                for phrase, key in commands.items():
                    if phrase in command_text:
                        matched_phrase = phrase
                        matched_key = key
                        break

                if matched_key is None or matched_phrase is None:
                    continue

                # Basic de-bounce to prevent repeated triggers.
                if (now - last_action_ts) < voice_cooldown and matched_key == last_command_key:
                    continue

                try:
                    pyautogui.press(matched_key)
                except Exception as e:
                    log_fn(f"Failed to execute voice command: {e}", kind="voice")
                    continue

                last_action_ts = time.time()
                last_command_key = matched_key

                # Update UI state (last command + slide direction for arrows).
                gesture_direction = None
                if matched_phrase == "next slide":
                    gesture_direction = "right"
                elif matched_phrase == "previous slide":
                    gesture_direction = "left"

                with state_lock:
                    state["voice"]["last_command"] = matched_phrase
                    if gesture_direction in ("left", "right"):
                        state["gesture"]["last_gesture"] = "next" if gesture_direction == "right" else "prev"

                log_fn(f"Voice detected: {matched_phrase}", kind="voice", gesture_direction=gesture_direction)

    finally:
        with state_lock:
            state["voice"]["running"] = False
            state["voice"]["paused"] = False


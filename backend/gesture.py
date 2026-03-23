from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional

import cv2
import pyautogui


def _count_extended_fingers(lm: Any, *, handedness: Optional[str] = None) -> int:
    """
    Count extended fingers using MediaPipe hand landmarks (21 points).
    """

    # Tip indices
    tip_ids = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    pip_ids = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}

    fingers = 0

    # Thumb (x-based, depends on handedness)
    thumb_tip = lm[tip_ids["thumb"]]
    thumb_ip = lm[pip_ids["thumb"]]
    if handedness is not None:
        # MediaPipe returns "Left"/"Right" relative to the image.
        if handedness.lower().startswith("right"):
            if thumb_tip.x > thumb_ip.x:
                fingers += 1
        else:
            if thumb_tip.x < thumb_ip.x:
                fingers += 1
    else:
        # Fallback: assume extended if thumb tip is further from wrist on x.
        wrist = lm[0]
        if abs(thumb_tip.x - wrist.x) > abs(thumb_ip.x - wrist.x):
            fingers += 1

    # Other fingers (y-based)
    for name in ("index", "middle", "ring", "pinky"):
        tip = lm[tip_ids[name]]
        pip = lm[pip_ids[name]]
        if tip.y < pip.y:
            fingers += 1

    return fingers


def run_gesture_detection(
    stop_event: Any,
    pause_event: Any,
    state: Dict[str, Any],
    state_lock: Any,
    log_fn: Callable[..., None],
    config: Dict[str, Any],
) -> None:
    """
    Gesture detection loop.
    Uses MediaPipe Hands (tasks if model exists, otherwise solutions fallback).
    """

    base_dir = Path(__file__).resolve().parent

    frame_skip = int(config.get("frame_skip", 3))
    # movement_threshold is interpreted as "normalized hand travel" threshold for swipe detection.
    movement_threshold = float(config.get("movement_threshold", 0.12))
    cooldown = float(config.get("cooldown", 0.8))
    cap_width, cap_height = config.get("resolution", [640, 480])
    palm_smoothing = int(config.get("palm_smoothing", 5))
    process_sleep = float(config.get("process_sleep", 0.005))

    # Try MediaPipe tasks API if the model exists; otherwise use solutions API.
    model_asset_path = config.get("model_asset_path", "hand_landmarker.task")
    model_path = base_dir / model_asset_path
    use_tasks_api = model_path.exists()

    mp = None
    landmarker = None
    hands = None
    try:
        import mediapipe as _mp

        mp = _mp
    except Exception as e:
        log_fn(f"Failed to import mediapipe: {e}", kind="gesture")
        with state_lock:
            state["gesture"]["running"] = False
        return

    # Open the camera as early as possible to avoid UI-perceived startup lag.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.read()  # warm-up

    if not cap.isOpened():
        log_fn("Camera failed to open", kind="gesture")
        with state_lock:
            state["gesture"]["running"] = False
        return

    if use_tasks_api:
        try:
            from mediapipe.tasks.python import vision  # type: ignore
            from mediapipe.tasks import python  # type: ignore

            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = vision.HandLandmarker
            HandLandmarkerOptions = vision.HandLandmarkerOptions
            VisionRunningMode = vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=2,
            )
            landmarker = HandLandmarker.create_from_options(options)
            log_fn("Using MediaPipe Tasks hand landmarker", kind="gesture")
        except Exception as e:
            landmarker = None
            use_tasks_api = False
            log_fn(f"Tasks API init failed, falling back to solutions: {e}", kind="gesture")
    if not use_tasks_api:
        try:
            hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
                model_complexity=1,
            )
            log_fn("Using MediaPipe Solutions Hands", kind="gesture")
        except Exception as e:
            log_fn(f"Solutions Hands init failed: {e}", kind="gesture")
            with state_lock:
                state["gesture"]["running"] = False
            return

    last_action_ts = 0.0
    last_action_type: Optional[str] = None
    last_palm_x_avg: Optional[float] = None
    frame_count = 0
    palm_x_history: Deque[float] = deque(maxlen=max(2, palm_smoothing))
    # Window-based swipe detection to reduce jitter (more reliable than frame-to-frame delta).
    swipe_window = int(config.get("swipe_window", 7))  # number of processed frames
    swipe_min_delta = float(config.get("swipe_min_delta", movement_threshold))
    swipe_x_window: Deque[float] = deque(maxlen=max(3, swipe_window))

    def _set_paused(paused: bool) -> None:
        with state_lock:
            state["gesture"]["paused"] = paused

    try:
        while not stop_event.is_set():
            if pause_event.is_set():
                _set_paused(True)
                time.sleep(0.05)
                continue

            _set_paused(False)

            frame_count += 1
            if frame_count % max(1, frame_skip) != 0:
                time.sleep(process_sleep)
                continue

            if cap is None:
                break

            success, frame = cap.read()
            if not success or frame is None:
                log_fn("Camera frame read failed", kind="gesture")
                time.sleep(0.2)
                continue

            # Mirror for intuitive gestures.
            frame = cv2.flip(frame, 1)

            # Ensure predictable size for downstream processing.
            if (frame.shape[1], frame.shape[0]) != (cap_width, cap_height):
                frame = cv2.resize(frame, (cap_width, cap_height))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            handedness = None
            hand_lms = None

            if use_tasks_api and landmarker is not None:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))
                if results.hand_landmarks:
                    hand_lms = results.hand_landmarks[0]
                    try:
                        if results.handedness:
                            handedness = results.handedness[0][0].category_name
                    except Exception:
                        handedness = None
            else:
                results = hands.process(rgb)
                if results.multi_hand_landmarks:
                    hand_lms = results.multi_hand_landmarks[0]
                    if results.multi_handedness:
                        try:
                            handedness = results.multi_handedness[0].classification[0].label
                        except Exception:
                            handedness = None

            if hand_lms is None:
                time.sleep(process_sleep)
                continue

            palm_x = float(hand_lms[0].x)  # wrist x (normalized)
            palm_x_history.append(palm_x)
            palm_x_avg = sum(palm_x_history) / len(palm_x_history)
            swipe_x_window.append(palm_x_avg)

            finger_count = _count_extended_fingers(hand_lms, handedness=handedness)
            now = time.time()

            # Stable swipe direction (avoid jitter) using window travel.
            swipe_delta = None
            if len(swipe_x_window) >= swipe_x_window.maxlen:
                swipe_delta = swipe_x_window[-1] - swipe_x_window[0]

            action_triggered = False
            # 4-finger "start presentation"
            if finger_count == 5 and (now - last_action_ts) > cooldown:
                pyautogui.press("home")
                last_action_type = "home"
                last_action_ts = now
                action_triggered = True
                log_fn("Gesture detected: Home Slide", kind="gesture")
            elif finger_count == 0 and (now - last_action_ts) > cooldown:
                pyautogui.press("end")
                last_action_type = "end"
                last_action_ts = now
                action_triggered = True
                log_fn("Gesture detected: Last Slide", kind="gesture")
            elif finger_count == 4 and (now - last_action_ts) > cooldown:
                pyautogui.press("f5")
                last_action_type = "start"
                last_action_ts = now
                action_triggered = True
                log_fn("Gesture detected: Start Presentation", kind="gesture")
            elif finger_count == 1 and (now - last_action_ts) > cooldown:
                pyautogui.press("esc")
                last_action_type = "stop"
                last_action_ts = now
                action_triggered = True
                log_fn("Gesture detected: Stop Presentation", kind="gesture")
            elif finger_count == 2 and (now - last_action_ts) > cooldown:
                pyautogui.press("right")
                last_action_type = "next"
                last_action_ts = now
                action_triggered = True
                log_fn("Gesture detected: Next Slide", kind="gesture", gesture_direction="right")
            elif finger_count == 3 and (now - last_action_ts) > cooldown:
                pyautogui.press("left")
                last_action_type = "prev"
                last_action_ts = now
                action_triggered = True
                log_fn("Gesture detected: Previous Slide", kind="gesture", gesture_direction="left")
            else:
                # Swipes: use window travel threshold and cooldown.
                # Allow swipe for "neutral" finger counts (2-3 fingers typically),
                # so fist/palm/start/stop poses don't interfere with slide nav.
                is_navigation_pose = finger_count in (2, 3)
                if is_navigation_pose and swipe_delta is not None:
                    if swipe_delta > swipe_min_delta and (now - last_action_ts) > cooldown:
                        pyautogui.press("right")
                        last_action_type = "next"
                        last_action_ts = now
                        action_triggered = True
                        log_fn("Gesture detected: Next Slide", kind="gesture", gesture_direction="right")
                        swipe_x_window.clear()
                    elif swipe_delta < -swipe_min_delta and (now - last_action_ts) > cooldown:
                        pyautogui.press("left")
                        last_action_type = "prev"
                        last_action_ts = now
                        action_triggered = True
                        log_fn("Gesture detected: Previous Slide", kind="gesture", gesture_direction="left")
                        swipe_x_window.clear()

            if action_triggered:
                with state_lock:
                    if last_action_type in ("next", "prev"):
                        state["gesture"]["last_direction"] = "right" if last_action_type == "next" else "left"
                    state["gesture"]["last_gesture"] = last_action_type

            last_palm_x_avg = palm_x_avg
            time.sleep(process_sleep)

    except Exception as e:
        log_fn(f"Gesture module crashed: {e}", kind="gesture")
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if hands is not None:
                hands.close()
        except Exception:
            pass
        try:
            if landmarker is not None:
                # tasks landmarker is a context-managed object in some versions,
                # so best-effort close.
                landmarker.close()
        except Exception:
            pass

        with state_lock:
            state["gesture"]["running"] = False
            state["gesture"]["paused"] = False
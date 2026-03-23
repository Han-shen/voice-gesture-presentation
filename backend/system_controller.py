from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional


JsonDict = Dict[str, Any]


@dataclass
class ModuleEvents:
    stop_event: threading.Event
    pause_event: threading.Event


class SystemController:
    """
    Thread-safe system controller for voice + gesture.
    Owns lifecycle, shared state, and a log queue for live UI updates.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        base_dir = Path(__file__).resolve().parent
        self._config_path = config_path or (base_dir / "config.json")

        self._start_stop_lock = threading.Lock()
        self._state_lock = threading.Lock()

        self._running: bool = False
        self._stop_event: Optional[threading.Event] = None

        self._gesture_events: Optional[ModuleEvents] = None
        self._voice_events: Optional[ModuleEvents] = None

        self._gesture_thread: Optional[threading.Thread] = None
        self._voice_thread: Optional[threading.Thread] = None

        self._log_queue: queue.Queue[JsonDict] = queue.Queue(maxsize=5000)

        self._state: JsonDict = {
            "overall_running": False,
            "gesture": {
                "running": False,
                "paused": False,
                "last_gesture": None,  # home|end|start|stop|next|prev
                "last_direction": None,  # left|right
            },
            "voice": {
                "running": False,
                "paused": False,
                "last_command": None,
            },
        }

        self._config: JsonDict = {}
        self._load_config()

    def _load_config(self) -> None:
        try:
            self._config = json.loads(self._config_path.read_text(encoding="utf-8"))
        except Exception:
            self._config = {}

    @property
    def config(self) -> JsonDict:
        return self._config

    def _set_state(self, mutator: Callable[[JsonDict], None]) -> None:
        with self._state_lock:
            mutator(self._state)

    def get_status(self) -> JsonDict:
        with self._state_lock:
            # shallow copy is enough for UI reads; nested dicts are rebuilt on write
            return json.loads(json.dumps(self._state))

    def _enqueue_log(self, payload: JsonDict) -> None:
        payload = dict(payload)
        payload.setdefault("ts", time.time())
        try:
            self._log_queue.put_nowait(payload)
        except queue.Full:
            try:
                _ = self._log_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._log_queue.put_nowait(payload)
            except queue.Full:
                # Drop if still full.
                return

    def log(self, message: str, *, kind: str = "system", gesture_direction: Optional[str] = None) -> None:
        """
        Queue a log message for the frontend.
        """

        self._enqueue_log(
            {
                "type": "log",
                "kind": kind,
                "message": message,
                "gesture_direction": gesture_direction,
            }
        )

        def _mut(state: JsonDict) -> None:
            state["last_feedback"] = message
            if gesture_direction in ("left", "right"):
                state["gesture"]["last_direction"] = gesture_direction

        self._set_state(_mut)

    def start(
        self,
        *,
        gesture_runner: Callable[..., None],
        voice_runner: Callable[..., None],
    ) -> JsonDict:
        with self._start_stop_lock:
            if self._running:
                # Return current status; caller can decide what HTTP code to use.
                return self.get_status()

            self._running = True
            self._stop_event = threading.Event()
            gesture_pause_event = threading.Event()
            voice_pause_event = threading.Event()
            gesture_pause_event.clear()
            voice_pause_event.clear()

            self._gesture_events = ModuleEvents(
                stop_event=self._stop_event,
                pause_event=gesture_pause_event,
            )
            self._voice_events = ModuleEvents(
                stop_event=self._stop_event,
                pause_event=voice_pause_event,
            )

            self._set_state(
                lambda s: s.update(
                    {
                        "overall_running": True,
                        "gesture": {
                            **s["gesture"],
                            "running": True,
                            "paused": False,
                        },
                        "voice": {
                            **s["voice"],
                            "running": True,
                            "paused": False,
                        },
                    }
                )
            )

            self._gesture_thread = threading.Thread(
                target=gesture_runner,
                args=(
                    self._gesture_events.stop_event,
                    self._gesture_events.pause_event,
                    self._state,
                    self._state_lock,
                    self.log,
                    self._config.get("gesture", {}),
                ),
                daemon=True,
            )
            self._voice_thread = threading.Thread(
                target=voice_runner,
                args=(
                    self._voice_events.stop_event,
                    self._voice_events.pause_event,
                    self._state,
                    self._state_lock,
                    self.log,
                    self._config.get("voice", {}),
                ),
                daemon=True,
            )

            self._gesture_thread.start()
            self._voice_thread.start()

            self.log("System started - Voice & Gesture active", kind="system")
            return self.get_status()

    def stop(self) -> JsonDict:
        with self._start_stop_lock:
            if not self._running:
                return self.get_status()

            # Signal threads to stop.
            if self._gesture_events:
                self._gesture_events.pause_event.set()
            if self._voice_events:
                self._voice_events.pause_event.set()
            if self._stop_event:
                self._stop_event.set()

            # Join (best-effort).
            t_g = self._gesture_thread
            t_v = self._voice_thread
            self._gesture_thread = None
            self._voice_thread = None

        # Join outside the lock so logs/threads aren't blocked.
        if t_g and t_g.is_alive():
            t_g.join(timeout=8)
        if t_v and t_v.is_alive():
            t_v.join(timeout=8)

        with self._state_lock:
            self._running = False
            self._state["overall_running"] = False
            self._state["gesture"]["running"] = False
            self._state["gesture"]["paused"] = False
            self._state["voice"]["running"] = False
            self._state["voice"]["paused"] = False

        self.log("System stopped", kind="system")
        return self.get_status()

    def pause_module(self, module: str, paused: bool) -> JsonDict:
        module = (module or "").strip().lower()
        with self._start_stop_lock:
            if not self._running:
                return self.get_status()

            if module == "gesture" and self._gesture_events:
                if paused:
                    self._gesture_events.pause_event.set()
                else:
                    self._gesture_events.pause_event.clear()
                self._set_state(lambda s: s["gesture"].update({"paused": paused}))
            elif module == "voice" and self._voice_events:
                if paused:
                    self._voice_events.pause_event.set()
                else:
                    self._voice_events.pause_event.clear()
                self._set_state(lambda s: s["voice"].update({"paused": paused}))
            else:
                return self.get_status()

        self.log(
            f"{module.capitalize()} {'paused' if paused else 'resumed'}",
            kind="system",
        )
        return self.get_status()

    def stream_logs(self):
        """
        SSE generator yielding events from the log queue.
        """
        import json as _json

        # Immediately notify the UI of connection (default "message" event).
        yield f"data: {_json.dumps({'type': 'log', 'kind': 'system', 'message': 'Connected', 'gesture_direction': None, 'ts': time.time()})}\n\n"

        while True:
            try:
                payload = self._log_queue.get(timeout=15)
                yield f"data: {_json.dumps(payload)}\n\n"
            except queue.Empty:
                # keep-alive comment for proxies
                yield ": keep-alive\n\n"


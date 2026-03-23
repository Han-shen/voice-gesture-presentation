import { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';

const API_BASE = 'http://127.0.0.1:8000';

function formatClock(tsSeconds) {
  try {
    return new Date(tsSeconds * 1000).toLocaleTimeString();
  } catch {
    return '';
  }
}

function App() {
  const [status, setStatus] = useState({
    overall_running: false,
    gesture: { running: false, paused: false, last_gesture: null, last_direction: null },
    voice: { running: false, paused: false, last_command: null },
    last_feedback: null,
  });
  const [logs, setLogs] = useState([]);
  const [connection, setConnection] = useState({ sse: false });
  const [errorText, setErrorText] = useState('');
  const [gestureDir, setGestureDir] = useState(null); // 'left' | 'right' | null

  const running = !!status.overall_running;
  const micActive = running && !!status.voice.running && !status.voice.paused;

  const chatScrollRef = useRef(null);
  const gestureClearTimerRef = useRef(null);
  const messagesById = useMemo(() => new Set(), []);

  const appendMessage = (item) => {
    const id = item.id;
    if (messagesById.has(id)) return;
    messagesById.add(id);
    setLogs((prev) => {
      const next = [...prev, item];
      return next.length > 80 ? next.slice(next.length - 80) : next;
    });

    if (item.gesture_direction === 'left' || item.gesture_direction === 'right') {
      setGestureDir(item.gesture_direction);
      if (gestureClearTimerRef.current) clearTimeout(gestureClearTimerRef.current);
      gestureClearTimerRef.current = setTimeout(() => setGestureDir(null), 900);
    }
  };

  useEffect(() => {
    let alive = true;

    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/status`, { cache: 'no-store' });
        if (!res.ok) return;
        const data = await res.json();
        if (!alive) return;
        setStatus(data);
      } catch {
        // ignore; SSE/logs might still be down too
      }
    };

    poll();
    const interval = setInterval(poll, 900);
    return () => {
      alive = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    let es = null;
    let closed = false;

    const connect = () => {
      if (es) {
        try {
          es.close();
        } catch {}
      }
      es = new EventSource(`${API_BASE}/logs/stream`);
      setConnection({ sse: true });

      es.onmessage = (ev) => {
        if (closed) return;
        try {
          const payload = JSON.parse(ev.data);
          if (!payload || payload.type !== 'log') return;
          const item = {
            id: `${payload.ts ?? 0}-${payload.kind ?? 'log'}-${payload.message ?? ''}`,
            ts: payload.ts ?? Math.floor(Date.now() / 1000),
            kind: payload.kind ?? 'system',
            message: payload.message ?? '',
            gesture_direction: payload.gesture_direction ?? null,
          };
          appendMessage(item);
        } catch {
          // ignore malformed payloads
        }
      };

      es.onerror = () => {
        setConnection({ sse: false });
        try {
          es.close();
        } catch {}
        es = null;
        if (closed) return;
        // Backoff reconnect
        setTimeout(() => {
          if (!closed) connect();
        }, 1200);
      };
    };

    connect();

    return () => {
      closed = true;
      if (es) {
        try {
          es.close();
        } catch {}
      }
      if (gestureClearTimerRef.current) clearTimeout(gestureClearTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!chatScrollRef.current) return;
    // Smooth scroll to latest message.
    chatScrollRef.current.scrollTo({ top: chatScrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [logs.length]);

  const startSystem = async () => {
    setErrorText('');
    try {
      const res = await fetch(`${API_BASE}/start`, { method: 'GET' });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || 'Failed to start');
      }
    } catch (e) {
      setErrorText(String(e?.message || e));
    }
  };

  const stopSystem = async () => {
    setErrorText('');
    try {
      const res = await fetch(`${API_BASE}/stop`, { method: 'GET' });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error || 'Failed to stop');
      }
    } catch (e) {
      setErrorText(String(e?.message || e));
    }
  };

  return (
    <div className="gp-app">
      <div className="gp-blurOrb gp-blurOrbA" />
      <div className="gp-blurOrb gp-blurOrbB" />

      <header className="gp-topbar">
        <div className="gp-title">
          <div className="gp-titleMain">Voice + Gesture</div>
          <div className="gp-titleSub">Presentation Control System</div>
        </div>

        <div className={`gp-statusPill ${running ? 'gp-statusRunning' : 'gp-statusStopped'}`}>
          <span className="gp-statusDot" />
          {running ? 'Running' : 'Stopped'}
        </div>
      </header>

      <main className="gp-container">
        <motion.section
          className="gp-controlCard"
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
        >
          <div className="gp-controlRow">
            <motion.button
              className="gp-btn gp-btnPrimary"
              onClick={startSystem}
              disabled={running}
              whileHover={{ scale: running ? 1 : 1.03 }}
              whileTap={{ scale: running ? 1 : 0.98 }}
            >
              Start
            </motion.button>

            <motion.button
              className="gp-btn gp-btnDanger"
              onClick={stopSystem}
              disabled={!running}
              whileHover={{ scale: !running ? 1 : 1.03 }}
              whileTap={{ scale: !running ? 1 : 0.98 }}
            >
              Stop
            </motion.button>
          </div>

          <div className="gp-feedbackRow">
            <div className="gp-micCard">
              <div className="gp-micTop">
                <div className="gp-micLabel">Microphone</div>
                <div className="gp-micText">{micActive ? 'Listening…' : 'Idle'}</div>
              </div>

              <div className={`gp-micIndicator ${micActive ? 'gp-micIndicatorActive' : ''}`}>
                <div className="gp-micCore" />
                <div className="gp-micWaves gp-micWaves1" />
                <div className="gp-micWaves gp-micWaves2" />
              </div>
            </div>

            <div className="gp-gestureCard">
              <div className="gp-gestureLabel">Gesture Feedback</div>
              <div className="gp-arrowStage" aria-live="polite">
                <AnimatePresence mode="wait">
                  {gestureDir === 'left' ? (
                    <motion.div
                      key="left"
                      className="gp-arrow gp-arrowLeft"
                      initial={{ opacity: 0, x: -12, scale: 0.96 }}
                      animate={{ opacity: 1, x: 0, scale: 1 }}
                      exit={{ opacity: 0, x: -10, scale: 0.95 }}
                      transition={{ duration: 0.18 }}
                    >
                      ←
                    </motion.div>
                  ) : null}

                  {gestureDir === 'right' ? (
                    <motion.div
                      key="right"
                      className="gp-arrow gp-arrowRight"
                      initial={{ opacity: 0, x: 12, scale: 0.96 }}
                      animate={{ opacity: 1, x: 0, scale: 1 }}
                      exit={{ opacity: 0, x: 10, scale: 0.95 }}
                      transition={{ duration: 0.18 }}
                    >
                      →
                    </motion.div>
                  ) : null}
                </AnimatePresence>

                {!gestureDir ? <div className="gp-arrowPlaceholder" /> : null}
              </div>
            </div>
          </div>

          {errorText ? <div className="gp-errorText">{errorText}</div> : null}
          <div className="gp-connectionHint">{connection.sse ? 'Live logs connected' : 'Connecting to logs…'}</div>
        </motion.section>

        <section className="gp-chatCard">
          <div className="gp-chatHeader">
            <div className="gp-chatTitle">Live Logs</div>
            <div className="gp-chatSubtitle">Voice + Gesture events</div>
          </div>

          <div className="gp-chatScroll" ref={chatScrollRef}>
            <AnimatePresence initial={false}>
              {logs.length === 0 ? (
                <motion.div
                  key="empty"
                  className="gp-emptyState"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  Press <b>Start</b> to begin.
                </motion.div>
              ) : (
                logs.map((m, idx) => (
                  <motion.div
                    key={m.id || idx}
                    className={`gp-bubble gp-bubble-${m.kind ?? 'system'}`}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 6 }}
                    transition={{ duration: 0.16 }}
                  >
                    <div className="gp-bubbleTop">
                      <span className="gp-bubbleKind">{(m.kind || 'system').toUpperCase()}</span>
                      <span className="gp-bubbleTime">{formatClock(m.ts)}</span>
                    </div>
                    <div className="gp-bubbleMsg">{m.message}</div>
                  </motion.div>
                ))
              )}
            </AnimatePresence>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;


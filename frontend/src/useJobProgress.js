import { useState, useEffect, useRef, useCallback } from 'react';

export default function useJobProgress(jobId) {
  const [progress, setProgress] = useState({
    status: null,
    stage: null,
    progress: 0,
    message: '',
    elapsed_seconds: 0,
    obj_size: 0,
    step_size: 0,
    stage_times: {},
  });
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);
  const retriesRef = useRef(0);
  const maxRetries = 3;

  const connect = useCallback(() => {
    if (!jobId) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${jobId}`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      retriesRef.current = 0;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.error) {
          setProgress((prev) => ({ ...prev, status: 'error', message: data.error }));
          return;
        }
        setProgress((prev) => ({
          ...prev,
          status: data.status,
          stage: data.stage,
          progress: data.progress,
          message: data.message,
          elapsed_seconds: data.elapsed_seconds || 0,
          obj_size: data.obj_size || 0,
          step_size: data.step_size || 0,
          stage_times: data.stage_times || {},
        }));
        if (data.status === 'done' || data.status === 'error') {
          ws.close();
        }
      } catch {
        // ignore malformed messages
      }
    };

    ws.onclose = () => {
      setConnected(false);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;
    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [jobId, connect]);

  // Reconnect logic (exponential backoff)
  useEffect(() => {
    if (connected || !jobId) return;
    if (retriesRef.current >= maxRetries) return;

    const delay = Math.min(1000 * Math.pow(2, retriesRef.current), 8000);
    const timer = setTimeout(() => {
      retriesRef.current += 1;
      connect();
    }, delay);

    return () => clearTimeout(timer);
  }, [connected, jobId, connect]);

  return { ...progress, connected };
}

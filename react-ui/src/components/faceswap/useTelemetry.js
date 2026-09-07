import { useEffect, useState } from 'react';
import { getJSON } from '../../api';

// Polls the backend system telemetry (GPU/VRAM/CPU/RAM/threads) for the HUD.
//
// `enabled` exists for callers that only want the numbers while a panel is
// actually on screen — App's header HUD is hidden by default, and a poll every
// three seconds for a panel nobody has opened is pure cost on a machine whose
// GPU is the point. The last value is kept when it goes false, so reopening the
// panel shows something immediately rather than a dash.
export default function useTelemetry(intervalMs = 3000, enabled = true) {
  const [telemetry, setTelemetry] = useState(null);

  useEffect(() => {
    if (!enabled) return undefined;
    const fetchTelemetry = async () => {
      try {
        const data = await getJSON('/api/system/telemetry');
        setTelemetry(data);
      } catch {
        // quiet fail
      }
    };
    fetchTelemetry();
    const id = setInterval(fetchTelemetry, intervalMs);
    return () => clearInterval(id);
  }, [intervalMs, enabled]);

  return telemetry;
}

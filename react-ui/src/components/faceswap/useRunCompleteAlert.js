import { useEffect, useRef, useState } from 'react';
import { playChime, notifyDesktop } from './utils';

// ── "Your render finished" ────────────────────────────────────────────────
// A chime always, plus an OS notification when the user has asked for one.
// Fires on the processing -> idle edge, so it cannot repeat while a run is in
// flight or announce a run that was already finished when the tab mounted.

export default function useRunCompleteAlert({ processing, notify }) {
  const [desktopAlerts, setDesktopAlerts] = useState(false);

  // notifyDesktop() silently does nothing unless permission is already granted,
  // so ask when the toggle is switched ON — otherwise enabling alerts is a
  // no-op for anyone who has not started a run first (start() also asks).
  const toggleDesktopAlerts = () => {
    const on = !desktopAlerts;
    setDesktopAlerts(on);
    if (on) {
      try {
        if (!('Notification' in window)) {
          notify('This browser has no desktop notifications', 'error');
        } else if (Notification.permission === 'denied') {
          notify('Desktop notifications are blocked for this site', 'error');
        } else if (Notification.permission === 'default') {
          Notification.requestPermission();
        }
      } catch { /* ignore */ }
    }
  };

  const prevProcessingRef = useRef(false);
  useEffect(() => {
    if (prevProcessingRef.current && !processing) {
      playChime();
      if (desktopAlerts) {
        notifyDesktop('Roop Unleashed Render Complete!', 'Your face swap processing run has finished.');
      }
    }
    prevProcessingRef.current = processing;
  }, [processing, desktopAlerts]);

  return { desktopAlerts, toggleDesktopAlerts };
}

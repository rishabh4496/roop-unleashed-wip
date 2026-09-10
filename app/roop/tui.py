"""Terminal presentation: severity, colour policy, and indeterminate stages.

This sits BESIDE procmgr_runtime's progress machinery; it does not replace it.
ChunkedProgress already handles the determinate case correctly, including the
fact that Pinokio's captured log cannot be rewritten in place, and nothing here
should tempt a caller away from it. Frames have a total: use ChunkedProgress.

What was missing is the rest of the terminal's output.

  * Status lines were classified by matching the message against keyword lists
    -- and the message interpolates user filenames, so the severity badge was
    decided by what the user named their file. Verified: `error_clip.mp4`
    printed [ERROR] at the moment its render STARTED, and `mistook_scene.mp4`
    printed [SUCCESS], because "took" is a substring of "mistook".

  * They were printed with a bare print(), which is precisely what bar_write
    exists to prevent -- see the note above ChunkedProgress, where one
    rewritten bar became one 451-character line per frame. bar_write is used 52
    times across five modules; core.py, which owns the app's primary status
    channel, used none of them.

  * The long stages BEFORE the frame counter starts moving -- model loads,
    TensorRT engine builds, ffmpeg extraction -- printed nothing at all, for
    minutes. procmgr_runtime's own words: a terminal that has said nothing for
    that long reads as a hang. That protection existed only inside the frame
    loop it was written for.
"""

import os
import shutil
import sys
import threading
import time

from roop.procmgr_runtime import (
    COLOR_ACCENT,
    COLOR_CYAN,
    COLOR_GRAY,
    COLOR_GREEN,
    COLOR_RESET,
    COLOR_YELLOW,
    bar_write,
)

COLOR_RED = "\033[91m"

# ── Severity ─────────────────────────────────────────────────────────────────
# A level is what the CALL SITE knows for free. It is never recovered from the
# prose, for the reason in the module docstring.
OK = "ok"
RUN = "run"
WARN = "warn"
ERR = "err"
INFO = "info"

# level -> (colour, unicode glyph, ascii glyph, plain tag)
_STYLE = {
    OK:   (COLOR_GREEN,  "✓", "+", "ok"),
    RUN:  (COLOR_ACCENT, "▸", ">", "run"),
    WARN: (COLOR_YELLOW, "!", "!", "warn"),
    ERR:  (COLOR_RED,    "✗", "x", "err"),
    INFO: (COLOR_CYAN,   "·", "-", "info"),
}


# ── Glyph policy ─────────────────────────────────────────────────────────────
def _glyphs_ok():
    """Whether the output stream can actually represent the box/braille glyphs.

    Tied to encodability rather than to isatty() or NO_COLOR, because that is
    the property that actually matters: run.py reconfigures the streams to
    UTF-8, but if that ever fails the console is left on the Windows ANSI
    codepage, where a single ✓ raises UnicodeEncodeError. bar_write already
    degrades such a line to replacement characters; testing up front lets us
    print something readable instead of a row of question marks.

    ROOP_ASCII=1 forces the plain set for a caller that wants it regardless.
    """
    if os.environ.get("ROOP_ASCII", "").strip().lower() in ("1", "on", "true"):
        return False
    enc = (getattr(sys.stderr, "encoding", None)
           or getattr(sys.stdout, "encoding", None) or "ascii")
    try:
        "✓▸✗·⠋".encode(enc)
        return True
    except (LookupError, UnicodeEncodeError):
        return False


_GLYPHS = _glyphs_ok()

# Separator for the non-TTY line format, matching procmgr_runtime's chunk line.
_SEP = " · " if _GLYPHS else " | "


def _glyph(style):
    """(colour, glyph, tag) for a _STYLE entry, picking the representable set."""
    color, uni, ascii_, tag = style
    return color, (uni if _GLYPHS else ascii_), tag


# ── Colour policy ────────────────────────────────────────────────────────────
def _resolve_color():
    """NO_COLOR > FORCE_COLOR > ROOP_COLOR > on.

    Defaulting to ON is deliberate, and is NOT the usual isatty() rule. This
    process's stdout is normally a pipe into Pinokio, whose console renders
    ANSI -- gating on isatty() would strip colour in exactly the place it
    works, which is why procmgr_runtime emits it unconditionally today. What
    was missing is the escape hatch for a genuine redirect to a file, so
    NO_COLOR (no-color.org) is honoured when merely PRESENT, whatever its
    value, as that spec requires.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    v = os.environ.get("ROOP_COLOR", "auto").strip().lower()
    if v in ("0", "off", "never", "false"):
        return False
    if v in ("1", "on", "always", "true"):
        return True
    return True


_COLOR = _resolve_color()


def color_enabled():
    return _COLOR


def paint(text, code):
    return f"{code}{text}{COLOR_RESET}" if _COLOR else str(text)


def _is_tty():
    # stderr, matching procmgr_runtime._stream_is_terminal: that is the stream
    # tqdm draws on, so it is the one that decides whether an in-place rewrite
    # means anything.
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


def _width(default=80):
    """Terminal width, re-queried per use rather than cached.

    That is the portable answer to a resize. SIGWINCH does not exist on
    Windows, and get_terminal_size() asks the OS each time, so a mid-render
    resize is picked up by the next line without a signal handler at all --
    the same approach dynamic_ncols=True already takes for the bars.
    """
    try:
        return max(40, shutil.get_terminal_size((default, 24)).columns)
    except Exception:
        return default


def _fmt_secs(s):
    try:
        s = float(s)
    except (TypeError, ValueError):
        return ""
    if s < 10:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{sec:02d}s" if m else f"{sec}s"


# ── Output arbitration ───────────────────────────────────────────────────────
# One spinner may own the last line of the terminal at a time. Any permanent
# line printed while it does has to erase it first, or the message lands on top
# of the spinner's half-drawn frame -- the same class of corruption bar_write
# prevents for the bars, which tqdm.write cannot know about here because the
# spinner is not a tqdm object.
_io_lock = threading.Lock()
_active_spinner = None


def _emit(line):
    """Write one permanent line, clearing any live spinner first."""
    with _io_lock:
        sp = _active_spinner
        if sp is not None:
            sp._erase_locked()
        bar_write(line)
        # The spinner redraws itself on its next tick (<=0.1s), so it does not
        # need to be restored here.


# ── Status lines ─────────────────────────────────────────────────────────────
def status(message, level=INFO):
    """One permanent status line, safe to call while a bar or spinner is live."""
    color, glyph, tag = _glyph(_STYLE.get(level, _STYLE[INFO]))
    # Leading newlines are stripped rather than printed. The colour prefix is
    # emitted BEFORE the message, so a message starting with "\n" put the badge
    # alone on one line and its text on the next. Callers wanting a visual
    # break should call blank_line().
    text = str(message).strip("\n")
    if _is_tty():
        _emit(f"  {paint(glyph, color)} {text}")
    else:
        _emit(f"{paint('[' + tag + ']', color)} {text}")


def blank_line():
    _emit("")


def step(name, detail="", seconds=None, level=OK):
    """A finished stage: the permanent single line a transient region collapses
    into. Columns are aligned so a run reads as a table rather than as prose."""
    color, glyph, tag = _glyph(_STYLE.get(level, _STYLE[OK]))
    dur = "" if seconds is None else _fmt_secs(seconds)
    detail = str(detail)
    if _is_tty():
        name_col = 22
        # 4 = two leading spaces + glyph + its trailing space.
        room = _width() - 4 - name_col - len(dur) - 2
        if room < 8:
            # Too narrow to align; fall back to a flowing line rather than
            # emitting negative padding or wrapping into an orphan row.
            bits = [b for b in (name, detail, dur) if b]
            _emit(f"  {paint(glyph, color)} " + "  ".join(bits))
            return
        body = f"{detail[:room]:<{room}}"
        _emit(f"  {paint(glyph, color)} {name:<{name_col}}"
              f"{paint(body, COLOR_CYAN)} {paint(dur, COLOR_GREEN)}")
    else:
        bits = [b for b in (name, detail, dur) if b]
        sep = paint(_SEP, COLOR_GRAY)
        _emit(f"{paint('[' + tag + ']', color)} " + sep.join(bits))


# ── Indeterminate stages ─────────────────────────────────────────────────────
class Spinner:
    """For work with no countable unit: model loads, TRT builds, ffmpeg.

    The determinate case belongs to ChunkedProgress and must stay there. This
    covers the minutes BEFORE a frame counter starts moving, where the terminal
    previously said nothing.

    On a TTY: one braille frame rewritten in place, erased on exit so the whole
    stage collapses into a single permanent step() line. Off a TTY: no
    animation whatsoever -- a heartbeat line at most every `every` seconds, so
    a captured log gets progress instead of ten frames a second of cursor
    codes. That split is the same one ChunkedProgress makes, for the same
    reason.

    Exceptions are never swallowed; a failing stage lands an [err] line
    carrying the exception and then propagates.
    """

    # Braille where the stream can carry it, a plain rotor where it cannot —
    # the ASCII set is four frames rather than ten, which is what a rotor
    # reads as at this interval.
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if _GLYPHS else "|/-\\"

    def __init__(self, name, detail="", every=15.0, interval=0.1):
        self.name = name
        self.detail = detail
        self.every = every
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._t0 = 0.0
        self._tty = _is_tty()
        self._drawn = False

    def __enter__(self):
        global _active_spinner
        self._t0 = time.perf_counter()
        with _io_lock:
            # Only one spinner owns the line. Nesting is a caller error, but it
            # must degrade to "the inner one animates" rather than to two
            # threads fighting over the same row.
            _active_spinner = self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        global _active_spinner
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with _io_lock:
            self._erase_locked()
            if _active_spinner is self:
                _active_spinner = None
        elapsed = time.perf_counter() - self._t0
        if exc_type is None:
            step(self.name, self.detail, elapsed, OK)
        else:
            step(self.name, f"{exc_type.__name__}: {exc}", elapsed, ERR)
        return False                      # never swallow

    def _erase_locked(self):
        """Caller holds _io_lock. Clears only the row this spinner drew."""
        if not (self._tty and self._drawn):
            return
        try:
            sys.stderr.write("\r\033[2K")
            sys.stderr.flush()
        except Exception:
            pass
        self._drawn = False

    def _draw(self, frame, elapsed):
        line = f"  {paint(frame, COLOR_ACCENT)} {self.name}"
        if self.detail:
            line += "  " + paint(self.detail, COLOR_GRAY)
        line += "  " + paint(_fmt_secs(elapsed), COLOR_GREEN)
        # Truncate against the CURRENT width so shrinking the window mid-stage
        # cannot wrap the line and leave an orphan row that \r cannot reach.
        # The budget is in visible characters; the escape sequences are not
        # printable, so they are added back on top of it.
        visible = _width()
        overhead = len(line) - len(_strip_ansi(line))
        _write(line[:visible + overhead])
        self._drawn = True

    def _run(self):
        i = 0
        last_beat = 0.0
        while not self._stop.is_set():
            elapsed = time.perf_counter() - self._t0
            if self._tty:
                with _io_lock:
                    if _active_spinner is self and not self._stop.is_set():
                        self._draw(self.FRAMES[i % len(self.FRAMES)], elapsed)
                i += 1
                self._stop.wait(self.interval)
            else:
                if elapsed - last_beat >= self.every:
                    last_beat = elapsed
                    sep = f" {_SEP} "
                    detail = f"{sep}{self.detail}" if self.detail else ""
                    _emit(f"{paint('[..]', COLOR_GRAY)} {self.name}{detail}"
                          f"{sep}{_fmt_secs(elapsed)} elapsed")
                self._stop.wait(0.5)


def _write(text):
    """Raw stderr write for the transient row. Never raises: an animation has
    no business killing a render, and the Windows ANSI codepage will refuse a
    braille glyph if run.py's UTF-8 reconfigure ever fails."""
    try:
        sys.stderr.write("\r\033[2K" + text)
        sys.stderr.flush()
    except Exception:
        pass


def _strip_ansi(text):
    out = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "\033":
            j = text.find("m", i)
            if j == -1:
                break
            i = j + 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)

"""Progress bars for worker processes, all drawn by the main process.

tqdm bars from separate processes overwrite each other, and so do the log
lines printed between them. Instead the workers send their bar updates and
log lines to the main process, which draws one bar per worker and prints the
log lines above them.

Main process::

    events = multiprocessing.get_context("spawn").Queue()
    with ProgressDisplay(events):
        ...  # start workers, handing them ``events``

Each worker::

    progress = WorkerProgress(events)
    silence_tqdm()
    bar = progress.bar("tracking", total=100)  # then bar.update(), bar.close()
    logger.add(progress.log)  # log lines show up above the bars
"""

import os
import threading

from tqdm import tqdm


def silence_tqdm() -> None:
    """Turn off every tqdm bar created in this process from now on.

    For worker processes: libraries draw their own bars, which would
    overwrite the ones the main process draws.
    """
    original = tqdm.__init__

    def init(self, *args, **kwargs):
        kwargs["disable"] = True
        original(self, *args, **kwargs)

    tqdm.__init__ = init


class RemoteBar:
    """A tqdm-like bar (``update``, ``close``) drawn by the main process."""

    def __init__(self, events, desc: str, total: int):
        self.events = events
        self._send("open", desc, total)

    def _send(self, kind: str, *args) -> None:
        self.events.put((kind, os.getpid(), *args))

    def update(self, n: int = 1) -> None:
        self._send("update", n)

    def close(self) -> None:
        self._send("close")


class WorkerProgress:
    """A worker's end of the ``events`` queue: its bars and its log lines."""

    def __init__(self, events):
        self.events = events

    def bar(self, desc: str, total: int) -> RemoteBar:
        return RemoteBar(self.events, desc, total)

    def log(self, message) -> None:
        """A loguru sink that prints ``message`` above the bars."""
        self.events.put(("log", os.getpid(), str(message)))


class ProgressDisplay:
    """Draws the bars and log lines the workers send over ``events``.

    Each worker gets its own row, in the order they first open a bar. Has to
    be used as a context manager, around the code that runs the workers.
    """

    def __init__(self, events):
        self.events = events
        self._rows = {}  # worker pid -> its row
        self._bars = {}  # worker pid -> its open bar
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "ProgressDisplay":
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self.events.put(None)  # all events sent before this are drawn first
        self._thread.join()

    def _run(self) -> None:
        while (event := self.events.get()) is not None:
            kind, pid, *args = event
            if kind == "log":
                tqdm.write(args[0], end="")
            elif kind == "open":
                self._close(pid)
                desc, total = args
                row = self._rows.setdefault(pid, len(self._rows))
                self._bars[pid] = tqdm(
                    total=total, desc=desc, position=row, leave=False
                )
            elif kind == "update" and pid in self._bars:
                self._bars[pid].update(args[0])
            elif kind == "close":
                self._close(pid)
        for pid in list(self._bars):
            self._close(pid)

    def _close(self, pid: int) -> None:
        if (bar := self._bars.pop(pid, None)) is not None:
            bar.close()

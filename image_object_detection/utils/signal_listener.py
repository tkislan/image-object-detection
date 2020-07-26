import signal
import threading


class SignalListener:
    def __init__(self, event: threading.Event):
        self._event = event
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, _, __):
        self._event.set()

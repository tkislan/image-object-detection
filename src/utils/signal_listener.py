import signal
from queue import Queue


class SignalListener:
    def __init__(self, q: Queue):
        self.__q = q
        signal.signal(signal.SIGINT, self.__signal_handler)
        signal.signal(signal.SIGTERM, self.__signal_handler)

    def __signal_handler(self, _, __):
        self.__q.put(None)

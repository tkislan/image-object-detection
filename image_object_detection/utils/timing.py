import time

def get_current_time_millis() -> int:
    return int(time.perf_counter() * 1000)
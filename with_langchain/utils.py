import signal
import logging
import functools

# ── Logging ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════
# TIMEOUT — decorator that enforces max execution time per tool
# ════════════════════════════════════════════════════════════════════════

class ToolTimeoutError(Exception):
    pass

def with_timeout(seconds: int):
    """Kills a tool function if it runs longer than `seconds`."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def _handler(signum, frame):
                raise ToolTimeoutError(
                    f"Tool '{func.__name__}' timed out after {seconds}s"
                )
            signal.signal(signal.SIGALRM, _handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
        return wrapper
    return decorator

# ════════════════════════════════════════════════════════════════════════
# ERROR RECOVERY — decorator that catches exceptions and returns them
# as strings so the LLM can read and adapt, rather than crashing
# ════════════════════════════════════════════════════════════════════════

def with_error_recovery(func):
    """Returns errors as readable strings instead of raising them."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ToolTimeoutError as e:
            msg = f"[TIMEOUT ERROR] {e}. Try a different input."
            logger.error(msg)
            return msg
        except ConnectionError as e:
            msg = f"[CONNECTION ERROR] Could not reach service: {e}"
            logger.error(msg)
            return msg
        except ValueError as e:
            msg = f"[INPUT ERROR] Bad input: {e}"
            logger.error(msg)
            return msg
        except Exception as e:
            msg = f"[UNEXPECTED ERROR] {type(e).__name__}: {e}"
            logger.error(msg)
            return msg
    return wrapper

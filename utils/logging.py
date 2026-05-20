import pigmento
from pigmento import pnt


_LOGGING_READY = False


def setup_logging():
    global _LOGGING_READY
    if _LOGGING_READY:
        return

    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False,
    )
    _LOGGING_READY = True

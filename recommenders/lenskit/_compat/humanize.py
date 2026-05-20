def metric(value, *args, **kwargs):
    return str(value)


def naturalsize(value, *args, **kwargs):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    idx = 0
    while abs(value) >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f"{value:.1f} {units[idx]}"

def to_float(x, default=0.0):
    """安全地转为 float"""
    try:
        return float(x)
    except Exception:
        return default

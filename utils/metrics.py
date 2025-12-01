def round_metrics(metrics, decimals=4):
    rounded = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            rounded[key] = round(value, decimals)
        elif isinstance(value, dict):
            rounded[key] = round_metrics(value, decimals)
        else:
            rounded[key] = value
    return rounded


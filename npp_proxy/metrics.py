"""BD-rate over the common task-metric interval."""

import numpy as np
from scipy.interpolate import PchipInterpolator


def rd_front(points):
    if any(not np.isfinite([rate, quality]).all() or rate <= 0 for rate, quality in points):
        raise ValueError('BD-rate requires finite metrics and positive bitrates')
    ordered = sorted(points, key=lambda pair: (-pair[1], pair[0]))
    frontier = []
    cheapest = float('inf')
    for rate, quality in ordered:
        if rate < cheapest and (not frontier or quality < frontier[-1][1]):
            frontier.append((rate, quality))
            cheapest = rate
    return frontier[::-1]


def bd_rate(anchor, candidate):
    anchor, candidate = rd_front(anchor), rd_front(candidate)
    result = {'percent': None, 'pchip_percent': None, 'interval': None,
              'anchor_points': len(anchor), 'candidate_points': len(candidate)}
    if min(len(anchor), len(candidate)) < 4:
        return {**result, 'reason': 'At least four nondominated points per curve are required'}
    low = max(anchor[0][1], candidate[0][1])
    high = min(anchor[-1][1], candidate[-1][1])
    if high <= low:
        return {**result, 'reason': 'No common quality interval'}

    def integrals(points):
        rate, quality = np.asarray(points).T
        log_rate = np.log(rate)
        cubic = np.polynomial.Polynomial.fit(quality, log_rate, deg=3).integ()
        pchip = PchipInterpolator(quality, log_rate)
        return np.array([cubic(high) - cubic(low), pchip.integrate(low, high)])

    percent, pchip_percent = 100 * np.expm1((integrals(candidate) - integrals(anchor)) / (high - low))
    return {**result, 'percent': float(percent), 'pchip_percent': float(pchip_percent),
            'interval': [low, high]}

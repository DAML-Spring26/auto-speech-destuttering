import numpy as np

def half_hann_fades(n: int):
    if n <= 0:
        return None, None
    t = np.linspace(0.0, np.pi, n, endpoint=False, dtype=np.float32)
    fade_in = 0.5 - 0.5 * np.cos(t)   # 0 → 1
    fade_out = 1.0 - fade_in         # 1 → 0
    return fade_in, fade_out


def merge_spans(spans, min_gap_s=0.02):
    if not spans:
        return []

    spans = sorted(spans)
    merged = [list(spans[0])]

    for s, e in spans[1:]:
        ps, pe = merged[-1]
        if s <= pe + min_gap_s:
            merged[-1][1] = max(pe, e)
        else:
            merged.append([s, e])

    return [(float(s), float(e)) for s, e in merged]


def complement_spans(remove_spans, total_dur_s):
    if not remove_spans:
        return [(0.0, float(total_dur_s))]

    keep = []
    cur = 0.0

    for s, e in remove_spans:
        if s > cur:
            keep.append((cur, s))
        cur = max(cur, e)

    if cur < total_dur_s:
        keep.append((cur, total_dur_s))

    return keep


def stitch_with_crossfade(y, sr: int, keep_segments_s, crossfade_ms: float = 20.0):
    if y.ndim == 1:
        y_ = y[:, None]
    else:
        y_ = y

    n_fade = int(sr * crossfade_ms / 1000.0)
    fade_in, fade_out = half_hann_fades(n_fade)

    out = None

    for s0, s1 in keep_segments_s:
        a = int(round(s0 * sr))
        b = int(round(s1 * sr))
        seg = y_[a:b]

        if seg.shape[0] == 0:
            continue

        if out is None:
            out = seg.astype(np.float32, copy=False)
            continue

        if n_fade <= 0 or out.shape[0] < n_fade or seg.shape[0] < n_fade:
            out = np.concatenate([out, seg], axis=0)
            continue

        # overlap-add crossfade
        tail = out[-n_fade:] * fade_out[:, None]
        head = seg[:n_fade] * fade_in[:, None]
        blended = tail + head

        out = np.concatenate([out[:-n_fade], blended, seg[n_fade:]], axis=0)

    if out is None:
        out = np.zeros((0, y_.shape[1]), dtype=np.float32)

    return out[:, 0] if y.ndim == 1 else out
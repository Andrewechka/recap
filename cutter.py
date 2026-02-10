class CutterConfig:
    def __init__(self, config):
        self.config = config

    def get(self, key, default=None):
        return self.config.get(key, default)


def _moving_avg_1d(data, window_size):
    weights = np.repeat(1.0, window_size) / window_size
    sma = np.convolve(data, weights, 'valid')
    return sma


def _diag_border_cuts(scene_data):
    cuts = []
    for i in range(len(scene_data) - 1):
        if scene_data[i] != scene_data[i + 1]:
            cuts.append(i)
    return cuts


def _color_change_cuts(scene_data, threshold=0.1):
    cuts = []
    for i in range(len(scene_data) - 1):
        if abs(scene_data[i] - scene_data[i + 1]) > threshold:
            cuts.append(i)
    return cuts


def split_scene_adaptive(scene_data):
    cuts = _diag_border_cuts(scene_data) + _color_change_cuts(scene_data)
    cuts = sorted(set(cuts))
    return cuts


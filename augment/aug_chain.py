import audiomentations as A


class AugChain:
    """
    统一增强管线，支持概率控制。
    Args:
        p (float): 每种增强的触发概率
    """

    def __init__(self, p: float = 0.3):
        self.aug = A.Compose([
            A.AddGaussianNoise(p=p),
            A.ApplyImpulseResponse(p=p),
            A.Mp3Compression(min_bitrate=32, max_bitrate=128, p=p),
            A.TimeStretch(min_rate=0.9, max_rate=1.1, p=p * 0.5),
        ])

    def __call__(self, wav, sr: int):
        """
        Args:
            wav: numpy array, shape (T,)
            sr:  采样率
        Returns:
            增强后的 numpy array, shape (T,)
        """
        return self.aug(samples=wav, sample_rate=sr)
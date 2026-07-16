"""Dataset loaders, kept lazy so their optional dependencies stay optional."""

__all__ = ["ImageSequenceDataset", "HM3DSemDataset", "CodaDataset"]


def __getattr__(name: str):
    if name == "ImageSequenceDataset":
        from .image_sequence import ImageSequenceDataset

        return ImageSequenceDataset
    if name == "HM3DSemDataset":
        from .hm3d_sem import HM3DSemDataset

        return HM3DSemDataset
    if name == "CodaDataset":
        from .coda import CodaDataset

        return CodaDataset
    raise AttributeError(name)

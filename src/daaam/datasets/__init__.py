"""Dataset public API with optional loaders imported only when requested."""

from .interfaces import BaseDataset, DatasetFrame

__all__ = [
    "BaseDataset",
    "DatasetFrame",
    "ImageSequenceDataset",
    "HM3DSemDataset",
    "CodaDataset",
]


def __getattr__(name: str):
    if name == "ImageSequenceDataset":
        from .loaders.image_sequence import ImageSequenceDataset

        return ImageSequenceDataset
    if name == "HM3DSemDataset":
        from .loaders.hm3d_sem import HM3DSemDataset

        return HM3DSemDataset
    if name == "CodaDataset":
        from .loaders.coda import CodaDataset

        return CodaDataset
    raise AttributeError(name)

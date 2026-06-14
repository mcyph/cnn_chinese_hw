"""PyTorch data pipeline for the handwriting recognizer.

This replaces the old ``TomoeDataset`` which pre-expanded every character into
~50 augmented rasters and serialised the whole thing to a 1.6 GB ``.npz``.
Instead we:

* parse the (small) stroke trajectories once and cache *those* to disk, and
* render + augment each sample on the fly inside the ``DataLoader`` workers.

This keeps the on-disk footprint tiny, lets each epoch see fresh augmentation,
and scales the number of augmented "virtual copies" with ``num_workers`` rather
than with RAM.

Which corpora are used is driven by ``DataConfig.license_group`` (see
``config.LICENSE_GROUPS``). To avoid co-mingling incompatibly-licensed data in
one set of weights, each group trains on mutually-compatible corpora and
validates on a held-out corpus from the *other* group:

* ``permissive`` -> trains on author + Tomoe (LGPL 2.1) + Make Me a Hanzi
  (Arphic PL); validates on held-out KanjiVG.
* ``ccbysa``     -> trains on author + KanjiVG (CC BY-SA 3.0); validates on
  held-out Tomoe.

The validation corpus is used only for early-stopping / calibration and never
enters the weights, so the permissive model's weights stay free of CC BY-SA. The
two models are combined at inference time (see ``recognizer.py``).
"""

import os
import pickle
import random
from collections import Counter

import torch
from torch.utils.data import Dataset

from cnn_chinese_hw.recognizer import config
from cnn_chinese_hw.parse_data.StrokeData import StrokeData
from cnn_chinese_hw.parse_data.iter_kanjivg_data import iter_kanjivg_data
from cnn_chinese_hw.parse_data.iter_makemeahanzi_data import (
    iter_makemeahanzi_data,
)
from cnn_chinese_hw.stroke_tools.HWDataAugmenter import HWStrokesAugmenter
from cnn_chinese_hw.stroke_tools.get_vertex import get_vertex
from cnn_chinese_hw.stroke_tools.points_normalized import points_normalized
from cnn_chinese_hw.gui.HandwritingRegister import HandwritingRegister


# Characters that should always be present (very common), even when building a
# small smoke-test subset. Preserved from the original project.
_ALWAYS_ADD = set(ord(c) for c in '目木日書的一是不了人我在有他这为之大来以个中上们')


def _stroke_key(ord_, strokes_list):
    """A deterministic, geometry-based de-duplication key.

    ``StrokeData.iter()`` yields the same ordinal once per script that contains
    it, each time with the full combined stroke set, so the same character's
    data would otherwise be added several times. Keying on the rounded point
    geometry (rather than on a namedtuple's incidental JSON encoding) makes the
    de-duplication robust to representation changes.
    """
    variants = tuple(
        tuple(tuple((round(x), round(y)) for x, y in stroke.LPoints)
              for stroke in variant)
        for variant in strokes_list
    )
    return (ord_, variants)


class StrokeStore:
    """Builds (and caches) the parsed stroke trajectories and the class list.

    Attributes
    ----------
    classes : list[int]
        Ordinals, indexed by class id (``classes[label] == ordinal``).
    train_samples / val_samples : list[(label, strokes, find_vertices)]
        ``strokes`` is ``[[(x, y), ...], ...]``; ``find_vertices`` tells the
        augmenter whether to run vertex simplification at render time.
    """

    def __init__(self, data_cfg: config.DataConfig, cache=True):
        self.cfg = data_cfg
        self.classes = []
        self._ord_to_label = {}

        cache_path = config.stroke_cache_path_for(data_cfg.license_group)
        if data_cfg.small_sample_only:
            cache_path = cache_path.replace('.pkl', '_sample.pkl')

        if cache and os.path.exists(cache_path):
            self._load(cache_path)
        else:
            self._build()
            if cache:
                self._save(cache_path)

    # -- public ----------------------------------------------------------
    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def class_counts(self):
        """Number of *base* training samples per class label (index = label).
        Used to build a class-balanced sampler."""
        counts = Counter(label for label, _, _ in self.train_samples)
        return [counts.get(i, 0) for i in range(self.num_classes)]

    # -- (de)serialisation ----------------------------------------------
    def _save(self, path):
        print(f"Saving stroke cache -> {path}")
        with open(path, 'wb') as f:
            pickle.dump({
                'classes': self.classes,
                'train_samples': self.train_samples,
                'val_samples': self.val_samples,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load(self, path):
        print(f"Loading stroke cache <- {path}")
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.classes = d['classes']
        self.train_samples = d['train_samples']
        self.val_samples = d['val_samples']
        self._ord_to_label = {o: i for i, o in enumerate(self.classes)}

    # -- building --------------------------------------------------------
    def _label_for(self, ord_, allow_new):
        if ord_ not in self._ord_to_label:
            if not allow_new:
                return None
            self._ord_to_label[ord_] = len(self.classes)
            self.classes.append(ord_)
        return self._ord_to_label[ord_]

    def _build(self):
        group = config.LICENSE_GROUPS[self.cfg.license_group]
        print(f"Parsing stroke trajectories for license group "
              f"'{self.cfg.license_group}' (train={group['train']}, "
              f"val={group['val']}; first run, cached)...")

        # Training: pool the group's compatible corpora (classes may be added).
        self.train_samples = list(self._iter_train_samples(group['train']))
        # Validation: the *other* group's held-out corpus, restricted to classes
        # that exist in training. It never enters the weights.
        self.val_samples = list(self._iter_val_samples(group['val']))

        print(f"  classes={self.num_classes} "
              f"train={len(self.train_samples)} val={len(self.val_samples)}")

    # -- per-corpus emitters --------------------------------------------------
    # Each yields (ordinal, drawings, find_vertices) where ``drawings`` is a list
    # of one-or-more drawings of that character, a drawing being [[(x, y), ...]].
    def _emit_author(self):
        register = HandwritingRegister()
        for ord_, strokes_list in register.get_D_strokes().items():
            drawings = []
            for strokes in strokes_list:
                strokes = points_normalized(strokes)
                drawings.append([get_vertex(s) for s in strokes])
            yield ord_, drawings, False

    def _emit_tomoe(self):
        # zh / zh-TW / ja, de-duplicated across scripts via rounded geometry.
        stroke_data = StrokeData()
        seen = set()
        for ord_, strokes_list in stroke_data.iter():
            key = _stroke_key(ord_, strokes_list)
            if key in seen:
                continue
            seen.add(key)
            drawings = [[s.LPoints for s in variant] for variant in strokes_list]
            yield ord_, drawings, False

    def _emit_makemeahanzi(self):
        # Arphic Public License; one synthetic stroke-median exemplar per char.
        try:
            it = iter_makemeahanzi_data()
        except FileNotFoundError as e:
            print(f"  [makemeahanzi] skipped: {e}")
            return
        for ord_, strokes_list in it:
            # Medians are sparse centre-lines already -> normalise, no re-vertex.
            yield ord_, [points_normalized(strokes_list)], False

    def _emit_kanjivg(self):
        # CC BY-SA. Bezier control points -> find_vertices=True at render time.
        for ord_, strokes in iter_kanjivg_data():
            yield ord_, [strokes], True

    def _emit_corpus(self, corpus):
        emit = {
            'author': self._emit_author,
            'tomoe': self._emit_tomoe,
            'makemeahanzi': self._emit_makemeahanzi,
            'kanjivg': self._emit_kanjivg,
        }[corpus]
        return emit()

    # -- sample iteration -----------------------------------------------------
    def _iter_train_samples(self, corpora):
        cfg = self.cfg
        for corpus in corpora:
            if corpus == 'makemeahanzi' and not cfg.use_makemeahanzi:
                continue
            # The author corpus is small and always kept in full; the large
            # corpora are truncated in smoke-test mode.
            truncate = corpus != 'author'
            for idx, (ord_, drawings, fv) in enumerate(self._emit_corpus(corpus)):
                if (truncate and cfg.small_sample_only
                        and ord_ not in _ALWAYS_ADD
                        and idx > cfg.small_sample_size):
                    continue
                label = self._label_for(ord_, allow_new=True)
                for strokes in drawings:
                    yield (label, strokes, fv)

    def _iter_val_samples(self, corpus):
        # Held-out corpus, restricted to classes that exist in training so the
        # label space matches the model's output (chars unseen in training are
        # skipped, exactly as in the original project).
        for ord_, drawings, fv in self._emit_corpus(corpus):
            label = self._label_for(ord_, allow_new=False)
            if label is None:
                continue
            for strokes in drawings:
                yield (label, strokes, fv)


class DirectMapDataset(Dataset):
    """Renders stroke samples to directMap tensors, with optional on-the-fly
    augmentation. Length is multiplied by ``copies`` so one DataLoader pass
    visits each base sample several times with different augmentation."""

    def __init__(self, samples, data_cfg: config.DataConfig,
                 augment: bool, copies: int = 1):
        self.samples = samples
        self.cfg = data_cfg
        self.augment = augment
        self.copies = max(1, copies)
        # Probability of returning the *unaugmented* render even in training,
        # so the model also sees clean strokes (helped in the original work).
        total = data_cfg.augmentations_per_sample + data_cfg.real_copies_per_sample
        self.real_prob = data_cfg.real_copies_per_sample / max(total, 1)

    def __len__(self):
        return len(self.samples) * self.copies

    def labels(self):
        """Label for every virtual index (aligned with ``__getitem__``)."""
        n = len(self.samples)
        return [self.samples[i % n][0] for i in range(len(self))]

    def __getitem__(self, idx):
        label, strokes, find_vertices = self.samples[idx % len(self.samples)]

        do_augment = self.augment
        if do_augment and random.random() < self.real_prob:
            do_augment = False

        aug = HWStrokesAugmenter(strokes, find_vertices=find_vertices)
        rastered = aug.raster_strokes(
            image_size=self.cfg.image_size,
            do_augment=do_augment,
        )
        # HxWxC uint8 -> CxHxW float in [0, 1]
        x = torch.from_numpy(rastered).permute(2, 0, 1).float().div_(255.0)
        return x, label


def build_datasets(data_cfg: config.DataConfig, cache=True):
    """Returns ``(store, train_ds, val_ds)``."""
    store = StrokeStore(data_cfg, cache=cache)
    train_ds = DirectMapDataset(
        store.train_samples, data_cfg, augment=True,
        copies=data_cfg.augmentations_per_sample,
    )
    val_ds = DirectMapDataset(
        store.val_samples, data_cfg, augment=False, copies=1,
    )
    return store, train_ds, val_ds


if __name__ == '__main__':
    # Quick sanity check of the data pipeline on a tiny subset.
    cfg = config.DataConfig(small_sample_only=True, small_sample_size=50)
    store, train_ds, val_ds = build_datasets(cfg, cache=False)
    x, y = train_ds[0]
    print("classes:", store.num_classes)
    print("sample tensor:", x.shape, x.dtype, float(x.min()), float(x.max()), "label:", y)

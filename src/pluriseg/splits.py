"""Dataset split management.

Splits live with the repo, one directory per dataset:

    data-splits/<dataset_name>/fold-<k>.yaml  -> {train/val/test: [[image, label, id], ...]}
    data-splits/<dataset_name>/manifest.yaml  -> seed, patterns, dedup record, ...

`make_train_test_split` (the CIS scheme): test = the official <test_dir>/,
val = a seeded val_fraction of <train_dir>/ (single fold-0.yaml, manifest
n_folds: 1). Deterministic from a seed; refuses to overwrite without
overwrite=True.
CLI: python -m pluriseg.cli make_train_test_split --config configs/splits/<name>.yaml
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import jsonargparse
import numpy as np
import yaml
from rich import print as rprint

from pluriseg.configs import DatasetConfig
from pluriseg.paths import SPLITS

EXTS = [".png", ".tif", ".tiff"]


def _extract_key(path: Path, regex: str | None) -> str:
    """Join all capture groups with '-' (or the stem without a regex).

    The key pairs images with labels and names the on-disk cache entry, so
    it must be unique per pair — e.g. w1/w2 variants of the same well stay
    distinct because the key includes both capture groups. The regex runs
    against the path relative to the dataset root, so directory parts like
    Cellpose/test/ can be captured too.
    """
    if regex is None:
        return path.stem
    m = re.search(regex, str(path))
    if not m:
        raise ValueError(f"regex {regex!r} did not match {str(path)}")
    return "-".join(m.groups()) if m.groups() else path.name


def list_pairs(
    root: Path,
    image_pattern: str,
    image_exclude_patterns: list[str] | str | None,
    label_pattern: str,
    label_exclude_patterns: list[str] | str | None,
    image_regex: str | None = None,
    label_regex: str | None = None,
) -> list[tuple[str, str, str]]:
    """Match images to labels by match_key (all regex capture groups joined
    with '-', or the stem without a regex).

    Raises if an image has no label. Labels with no image are warned
    about and dropped. Returns sorted [(image, label, id)] paths relative
    to root; id (the match key) is unique per pair and names the on-disk
    cache entry.
    """

    if isinstance(image_exclude_patterns, str):
        image_exclude_patterns = [image_exclude_patterns]

    if isinstance(label_exclude_patterns, str):
        label_exclude_patterns = [label_exclude_patterns]

    root = Path(root)
    images = set(root.glob(image_pattern))
    if image_exclude_patterns:
        n_before = len(images)
        excluded = set()
        for pat in image_exclude_patterns:
            excluded = excluded.union(set(root.glob(pat)))
        images = images - excluded
        rprint(f"[dim]{root}[/] image_exclude_patterns={image_exclude_patterns!r} "
               f"excluded {n_before - len(images)}/{n_before} images")

    labels = set(root.glob(label_pattern))
    if label_exclude_patterns:
        n_before = len(labels)
        excluded = set()
        for pat in label_exclude_patterns:
            excluded = excluded.union(set(root.glob(pat)))
        labels = labels - excluded
        rprint(f"[dim]{root}[/] label_exclude_patterns={label_exclude_patterns!r} "
               f"excluded {n_before - len(labels)}/{n_before} labels")

    rprint(
        f"[dim]{root}[/] image_pattern={image_pattern!r} -> {len(images)} images")
    rprint(
        f"[dim]{root}[/] label_pattern={label_pattern!r} -> {len(labels)} labels")

    def collect(paths: set[Path], regex: str | None) -> dict[str, Path]:
        """Path per match_key; raises on duplicate keys, which would
        otherwise make the image->label match depend on arbitrary set order.
        The regex runs against the path relative to root, so directory
        parts like Cellpose/test/ can be captured."""
        m: dict[str, Path] = {}
        for p in paths:
            k = _extract_key(p.relative_to(root), regex)
            if k in m:
                raise ValueError(
                    f"key collision: {m[k]} and {p} both map to key {k!r}")
            m[k] = p
        return m

    images = collect(images, image_regex)
    labels = collect(labels, label_regex)

    if not images:
        raise FileNotFoundError(
            f"no images matched {image_pattern!r} in {root}")

    missing_labels = set(images) - set(labels)
    if missing_labels:
        sample = [str(images[k].relative_to(root))
                  for k in sorted(missing_labels)[:5]]
        maskish = [s for s in missing_labels if re.search(
            r"(_|-|\b)?(mask|label|gt)(s)?(_|-|\b|\.|\[)", s, re.IGNORECASE)]
        hint = ""
        if maskish:
            hint = (f"\n{len(maskish)}/{len(missing_labels)} of the unmatched "
                    f"images look like ground-truth masks/labels themselves "
                    f"({', '.join(maskish[:3])}); is image_pattern "
                    f"{image_pattern!r} too broad? "
                    f"Consider using image_exclude_patterns.")
        if len(missing_labels) == len(images):
            raise FileNotFoundError(
                f"all {len(images)} images are missing a label in {root}; "
                f"likely a bad label_pattern {label_pattern!r} or "
                f"label_regex {label_regex!r}\nsample images:\n"
                + "\n".join(sample) + hint)

        raise FileNotFoundError(
            f"no labels for {len(missing_labels)}/{len(images)} images\n"
            f"sample images:\n" + "\n".join(sample) + hint)

    extra_labels = set(labels) - set(images)
    for k in extra_labels:
        rprint(
            f"[yellow bold]warning:[/] unused label {labels[k].relative_to(root)}")
    if extra_labels:
        rprint(f"[yellow bold]warning:[/] {len(extra_labels)} unused labels")

    pairs = sorted(
        (str(images[k].relative_to(root)), str(labels[k].relative_to(root)), k)
        for k in images
    )
    rprint(f"[dim]{root}[/] [bold]{len(pairs)}[/] image/label pairs matched")
    return pairs


def _dedup_pairs(
    pairs: list[tuple[str, str, str]],
    root: Path,
) -> tuple[list[tuple[str, str, str]], list[dict]]:
    """Drop byte-identical images: md5 over raw image bytes, keep the first
    occurrence in the (already sorted) pair order. A colliding pair survives
    only if its label is also byte-identical; otherwise warn and keep the
    first pair. Pairs arrive sorted, so 'first' is deterministic.

    Returns (kept pairs, dropped entries) — each dropped entry records the
    dropped and kept image/label/id plus whether the labels were identical.
    """
    import hashlib
    kept: list[tuple[str, str, str]] = []
    dropped: list[dict] = []
    seen: dict[str, tuple[str, str, str]] = {}
    for pair in pairs:
        h = hashlib.md5((root / pair[0]).read_bytes()).hexdigest()
        prev = seen.get(h)
        if prev is None:
            seen[h] = pair
            kept.append(pair)
            continue
        same_label = (
            (root / prev[1]).read_bytes() == (root / pair[1]).read_bytes())
        if same_label:
            rprint(f"[dim]dedup: dropped {pair[0]} (duplicate of {prev[0]})")
        else:
            rprint(
                "[yellow bold]warning:[/] dedup: {p[0]} duplicates image of "
                "{q[0]} but labels differ; keeping {q[0]}".format(
                    p=pair, q=prev))
        dropped.append({
            "dropped_image": pair[0],
            "dropped_label": pair[1],
            "dropped_id": pair[2],
            "kept_image": prev[0],
            "kept_label": prev[1],
            "kept_id": prev[2],
            "labels_identical": same_label,
        })
    return kept, dropped


def make_train_test_split(
    dataset_root: Path,
    dataset_name: str | None,
    splits_dir: Path | None,
    image_pattern: str,
    label_pattern: str,
    image_exclude_patterns: list[str] | str | None = None,
    label_exclude_patterns: list[str] | str | None = None,
    image_regex: str | None = None,
    label_regex: str | None = None,
    train_dir: str = "train",
    test_dir: str = "test",
    val_fraction: float = 0.1,
    seed: int = 0,
    overwrite: bool = False,
    dedup: bool = False,
) -> list[Path]:
    """Split a standard train/<test>/ layout: test = <test_dir>, val =
    val_fraction of <train_dir> (chosen with seed), train = the rest.

    One pattern/regex set serves both directories: image_pattern is globbed
    as f"{train_dir}/{image_pattern}" and f"{test_dir}/{image_pattern}"; the
    regexes run against paths relative to dataset_root (so they may capture
    the directory part, but need not). The match key is relative to the
    split directory, so train and test ids can collide — test cache ids get
    a 'test-' prefix to keep them unique. Val entries share the train cache
    ids (same image, same cache entry).

    Writes fold-0.yaml (train/val/test) + manifest.yaml with n_folds: 1;
    the data config must set n_folds: 1 for load_split's manifest check.
    Refuses to overwrite unless overwrite=True. With dedup=True, images
    byte-identical across the two directories are dropped from the train
    side (the official test set stays intact) and listed in the manifest.
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    if splits_dir is None:
        splits_dir = SPLITS
        rprint(
            f"splits_dir not set: using default [bold]{splits_dir}[/] ([dim]{splits_dir.resolve()})[/]")

    if dataset_name is None:
        dataset_name = dataset_root.name
        rprint(f"Automatically set dataset_name to {repr(dataset_name)}")

    def part_pairs(dir_name: str) -> list[tuple[str, str, str]]:
        pairs = list_pairs(
            root=dataset_root,
            image_pattern=f"{dir_name}/{image_pattern}",
            image_exclude_patterns=image_exclude_patterns,
            label_pattern=f"{dir_name}/{label_pattern}",
            label_exclude_patterns=label_exclude_patterns,
            image_regex=image_regex,
            label_regex=label_regex,
        )
        if dedup:
            n_before = len(pairs)
            pairs, _ = _dedup_pairs(pairs, dataset_root)
            if len(pairs) < n_before:
                rprint(
                    f"[dim]dedup: {n_before} -> {len(pairs)} pairs in {dir_name}/")
        return pairs

    train_pairs = part_pairs(train_dir)
    test_pairs = part_pairs(test_dir)
    dropped_leaks = []
    if dedup:
        # cross-part leak: same image bytes on both sides. Policy: keep the
        # official test set intact and drop the train-side copies (they may
        # even carry different labels); recorded in the manifest.
        def hash_map(pairs):
            return {
                hashlib.md5((dataset_root / image).read_bytes()).hexdigest():
                    (image, label, cid)
                for image, label, cid in pairs
            }
        train_h = hash_map(train_pairs)
        shared = train_h.keys() & hash_map(test_pairs).keys()
        dropped_leaks = []
        if shared:
            leaked_pairs = {train_h[h] for h in shared}
            train_pairs = [p for p in train_pairs if p not in leaked_pairs]
            dropped_leaks = [
                {"dropped_image": train_h[h][0],
                 "dropped_label": train_h[h][1],
                 "dropped_id": train_h[h][2]}
                for h in sorted(shared)
            ]
            rprint(f"[yellow bold]warning:[/] {len(shared)} train image(s) "
                   f"byte-identical to {test_dir}/ copies; dropping them "
                   f"(train/test leak)")

    # path overlap is a leak even when bytes differ
    train_imgs = {(dataset_root / i).resolve() for i, _, _ in train_pairs}
    test_imgs = {(dataset_root / i).resolve() for i, _, _ in test_pairs}
    overlap = train_imgs & test_imgs
    if overlap:
        raise ValueError(
            f"{len(overlap)} image path(s) appear in both {train_dir}/ and "
            f"{test_dir}/, e.g. {next(iter(overlap))}")

    # test ids are keyed relative to test_dir, so they can collide with
    # train ids (e.g. train/0000 vs test/0000); prefix them for the cache
    test_pairs = [(i, l, f"test-{cid}") for i, l, cid in test_pairs]

    n_val = int(round(val_fraction * len(train_pairs)))
    if n_val == 0:
        raise ValueError(
            f"val_fraction={val_fraction} of {len(train_pairs)} train pairs "
            f"yields an empty val set")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_pairs))
    val = sorted(train_pairs[i] for i in order[:n_val])
    train = sorted(train_pairs[i] for i in order[n_val:])

    splits_dir = splits_dir / dataset_name
    splits_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(splits_dir.glob("fold-*.yaml")):
        raise FileExistsError(
            f"{splits_dir} already holds fold files; pass --overwrite to regenerate")

    fold = {"train": train, "val": val, "test": test_pairs}
    path = splits_dir / "fold-0.yaml"
    with open(path, "w", encoding='utf-8') as f:
        yaml.safe_dump(fold, f, sort_keys=False)

    manifest = {
        "dataset_name": dataset_name,
        "scheme": (f"train-test: val={val_fraction} of {train_dir}/ (seed), "
                   f"test={test_dir}/"),
        "n_folds": 1,
        "seed": seed,
        "val_fraction": val_fraction,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test_pairs),
        "n_pairs": len(train) + len(val) + len(test_pairs),
        "dedup": dedup,
        "dropped_leaks": dropped_leaks,
        "image_pattern": image_pattern,
        "label_pattern": label_pattern,
        "image_regex": image_regex,
        "label_regex": label_regex,
        "image_exclude_patterns": image_exclude_patterns,
        "label_exclude_patterns": label_exclude_patterns,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    with open(splits_dir / "manifest.yaml", "w", encoding='utf-8') as f:
        yaml.dump(manifest, f, indent=2)
    return [path]


def load_split(
    cfg: DatasetConfig,
) -> tuple[str, dict[str, list[tuple[Path, Path, str]]]]:
    """Resolve the configured split into {train, val, test} pair lists.

    Loads fold-<cfg.fold>.yaml from cfg.splits_dir / <split_name> /, where
    split_name is required (a split name may differ from the dataset root's
    dir name). The manifest next to it must declare the same n_folds as
    cfg. Returns (dataset_name from the manifest, pairs resolved to
    absolute Paths against cfg.root, each with its cache id).
    """

    dataset_root = Path(cfg.root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {dataset_root}")
    if not cfg.split_name:
        raise ValueError("cfg.split_name is empty")

    splits_dir = Path(cfg.splits_dir) / cfg.split_name
    split_file = splits_dir / f"fold-{cfg.fold}.yaml"
    if not split_file.exists():
        # e.g. a checkpoint trained on the cluster carries its pasteur
        # splits_dir; fall back to the repo-local data-splits/ copy (manifest
        # must come from the same directory as the fold file)
        repo_dir = SPLITS / cfg.split_name
        repo_file = repo_dir / f"fold-{cfg.fold}.yaml"
        if repo_file.exists():
            rprint(
                f"Did not find {split_file} here, but {repo_file}: replacing")
            splits_dir = repo_dir
            split_file = repo_file
        else:
            raise FileNotFoundError(
                f"no fold-{cfg.fold}.yaml in {splits_dir} nor {repo_dir}")
    with open(split_file, encoding='utf-8') as f:
        split = yaml.safe_load(f)
    for part in ("train", "val", "test"):
        if part not in split:
            raise KeyError(f"{split_file} is missing the {part!r} section")

    manifest_file = splits_dir / "manifest.yaml"
    with open(manifest_file, encoding='utf-8') as f:
        manifest = yaml.safe_load(f)
    if manifest["n_folds"] != cfg.n_folds:
        raise ValueError(
            f"fold-{cfg.fold}.yaml belongs to a {manifest['n_folds']}-fold "
            f"scheme but cfg.n_folds={cfg.n_folds} ({manifest_file})")
    dataset_name = manifest['dataset_name']
    return dataset_name, {
        part: [
            (dataset_root / image, dataset_root / label, cache_id)
            for image, label, cache_id in ids
        ]
        for part, ids in split.items()
    }

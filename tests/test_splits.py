import yaml


def test_train_test_split_drops_leaked_train_copies(tmp_path):
    """A train image byte-identical to a test image (with a different label)
    is dropped from the train side; the official test set stays intact."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from pluriseg.splits import make_train_test_split

    root = tmp_path / "ds"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    for i, img in enumerate((b"a", b"b", b"c")):
        (root / "train" / f"{i:04d}_img.tif").write_bytes(img)
        (root / "train" / f"{i:04d}_masks.tif").write_bytes(b"lbl" + img)
    # test image byte-identical to train/0002, but a different label
    (root / "test" / "0000_img.tif").write_bytes(b"c")
    (root / "test" / "0000_masks.tif").write_bytes(b"other")

    written = make_train_test_split(
        dataset_root=root,
        dataset_name="ds",
        splits_dir=tmp_path / "splits",
        image_pattern="*_img.tif",
        label_pattern="*_masks.tif",
        image_regex=r"([0-9]+)_img\.tif",
        label_regex=r"([0-9]+)_masks\.tif",
        val_fraction=0.5,
        dedup=True,
    )
    fold = yaml.safe_load(open(written[0]))
    assert [p[2] for p in fold["test"]] == ["test-0000"]
    train_ids = {p[2] for p in fold["train"]} | {p[2] for p in fold["val"]}
    assert train_ids == {"0000", "0001"}
    manifest = yaml.safe_load(
        open(tmp_path / "splits" / "ds" / "manifest.yaml"))
    assert manifest["dropped_leaks"][0]["dropped_id"] == "0002"

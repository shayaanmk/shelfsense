import pandas as pd
import pytest

from forecasting import prepare_data

DAY_COLS = ["d_1", "d_2", "d_3"]


def sales_frame() -> pd.DataFrame:
    """Two FOODS/CA_1 SKUs with different volumes, plus rows that must be filtered out."""
    rows = [
        ("FOODS_1_001_CA_1_evaluation", "FOODS_1_001", "FOODS_1", "FOODS", "CA_1", "CA", 5, 5, 5),
        ("FOODS_1_002_CA_1_evaluation", "FOODS_1_002", "FOODS_1", "FOODS", "CA_1", "CA", 1, 0, 2),
        ("FOODS_1_003_CA_1_evaluation", "FOODS_1_003", "FOODS_1", "FOODS", "CA_1", "CA", 0, 0, 1),
        ("FOODS_1_004_CA_2_evaluation", "FOODS_1_004", "FOODS_1", "FOODS", "CA_2", "CA", 9, 9, 9),
        ("HOBBIES_1_001_CA_1_evaluation", "HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "CA_1", "CA", 7, 7, 7),
    ]
    columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", *DAY_COLS]
    return pd.DataFrame(rows, columns=columns)


def calendar_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "d": DAY_COLS,
            "date": ["2011-01-29", "2011-01-30", "2011-01-31"],
            "wm_yr_wk": [11101, 11101, 11101],
            "wday": [1, 2, 3],
            "month": [1, 1, 1],
            "year": [2011, 2011, 2011],
            "event_name_1": [None, "SuperBowl", None],
            "event_type_1": [None, "Sporting", None],
            "snap_CA": [0, 1, 1],
            "snap_TX": [0, 0, 1],
            "snap_WI": [1, 1, 0],
        }
    )


@pytest.fixture
def m5_dirs(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    monkeypatch.setattr(prepare_data, "RAW", raw)
    monkeypatch.setattr(prepare_data, "PROCESSED", processed)
    return raw, processed


def write_inputs(raw, sales=None, sales_name="sales_train_evaluation.csv"):
    (sales if sales is not None else sales_frame()).to_csv(raw / sales_name, index=False)
    calendar_frame().to_csv(raw / "calendar.csv", index=False)
    pd.DataFrame({"store_id": ["CA_1"], "item_id": ["FOODS_1_001"]}).to_csv(
        raw / "sell_prices.csv", index=False
    )


class TestFindSalesFile:
    def test_prefers_evaluation_over_validation(self, m5_dirs):
        raw, _ = m5_dirs
        (raw / "sales_train_validation.csv").touch()
        (raw / "sales_train_evaluation.csv").touch()
        assert prepare_data._find_sales_file().name == "sales_train_evaluation.csv"

    def test_falls_back_to_validation(self, m5_dirs):
        raw, _ = m5_dirs
        (raw / "sales_train_validation.csv").touch()
        assert prepare_data._find_sales_file().name == "sales_train_validation.csv"

    def test_missing_sales_file_raises(self, m5_dirs):
        with pytest.raises(FileNotFoundError, match="Download the M5 dataset"):
            prepare_data._find_sales_file()


class TestMain:
    def test_writes_long_parquet_and_sku_list(self, m5_dirs, monkeypatch, capsys):
        raw, processed = m5_dirs
        write_inputs(raw)
        monkeypatch.setitem(prepare_data.CONFIG, "n_skus", 2)

        prepare_data.main()

        long = pd.read_parquet(processed / "sales_long.parquet")
        # top 2 SKUs of the FOODS/CA_1 slice, one row per (item, day)
        assert sorted(long["item_id"].unique()) == ["FOODS_1_001", "FOODS_1_002"]
        assert len(long) == 2 * len(DAY_COLS)
        assert long["date"].dtype.kind == "M"
        assert long["date"].min() == pd.Timestamp("2011-01-29")
        # rows are sorted by (item_id, date)
        assert long[["item_id", "date"]].equals(
            long.sort_values(["item_id", "date"]).reset_index(drop=True)[["item_id", "date"]]
        )
        skus = (processed / "skus.txt").read_text(encoding="utf-8").splitlines()
        assert skus == ["FOODS_1_001", "FOODS_1_002"]
        assert "Kept 2 SKUs" in capsys.readouterr().out

    def test_snap_column_is_selected_by_store_and_renamed(self, m5_dirs, monkeypatch):
        raw, processed = m5_dirs
        write_inputs(raw)
        monkeypatch.setitem(prepare_data.CONFIG, "n_skus", 1)

        prepare_data.main()

        long = pd.read_parquet(processed / "sales_long.parquet")
        assert "snap" in long.columns
        assert not [c for c in long.columns if c.startswith("snap_")]
        assert long.sort_values("date")["snap"].tolist() == [0, 1, 1]
        assert long.sort_values("date")["event_name_1"].tolist()[1] == "SuperBowl"

    def test_units_survive_the_wide_to_long_melt(self, m5_dirs, monkeypatch):
        raw, processed = m5_dirs
        write_inputs(raw)
        monkeypatch.setitem(prepare_data.CONFIG, "n_skus", 2)

        prepare_data.main()

        long = pd.read_parquet(processed / "sales_long.parquet")
        by_item = long.groupby("item_id")["units"].sum()
        assert by_item["FOODS_1_001"] == 15
        assert by_item["FOODS_1_002"] == 3

    def test_other_store_scoping(self, m5_dirs, monkeypatch):
        raw, processed = m5_dirs
        write_inputs(raw)
        monkeypatch.setitem(prepare_data.CONFIG, "store", "CA_2")
        monkeypatch.setitem(prepare_data.CONFIG, "n_skus", 5)

        prepare_data.main()

        long = pd.read_parquet(processed / "sales_long.parquet")
        assert long["item_id"].unique().tolist() == ["FOODS_1_004"]

    def test_empty_slice_raises(self, m5_dirs, monkeypatch):
        raw, _ = m5_dirs
        write_inputs(raw)
        monkeypatch.setitem(prepare_data.CONFIG, "store", "WI_3")

        with pytest.raises(ValueError, match="No rows for category=FOODS store=WI_3"):
            prepare_data.main()

    @pytest.mark.parametrize("missing", ["calendar.csv", "sell_prices.csv"])
    def test_missing_companion_csv_raises(self, m5_dirs, missing):
        raw, _ = m5_dirs
        write_inputs(raw)
        (raw / missing).unlink()

        with pytest.raises(FileNotFoundError, match="unzip all M5 CSVs"):
            prepare_data.main()

    def test_creates_processed_directory(self, m5_dirs, monkeypatch):
        raw, processed = m5_dirs
        write_inputs(raw)
        monkeypatch.setitem(prepare_data.CONFIG, "n_skus", 1)
        assert not processed.exists()

        prepare_data.main()

        assert processed.is_dir()

from __future__ import annotations

from car_price_prediction.data import download
from car_price_prediction.config import AppSettings


class FakeKaggleApi:
    def __init__(self):
        self.calls = []

    def dataset_download_files(
        self,
        dataset,
        path=None,
        force=False,
        quiet=True,
        unzip=False,
    ):
        self.calls.append(
            {
                "dataset": dataset,
                "path": path,
                "force": force,
                "quiet": quiet,
                "unzip": unzip,
            }
        )
        download.config.RAW_DATA_PATH.write_text("fake csv", encoding="utf-8")


def test_download_dataset_uses_kaggle_python_api(monkeypatch, tmp_path):
    fake_api = FakeKaggleApi()

    monkeypatch.setattr(download.config, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(download.config, "RAW_DATA_PATH", tmp_path / "Car_sale_ads.csv")
    monkeypatch.setattr(
        download.config,
        "settings",
        AppSettings(kaggle_api_token="test-token"),
    )
    monkeypatch.setattr(download, "build_kaggle_api", lambda: fake_api)

    download.download_dataset(force=True)

    assert fake_api.calls == [
        {
            "dataset": download.config.KAGGLE_DATASET_ID,
            "path": str(tmp_path),
            "force": True,
            "quiet": False,
            "unzip": True,
        }
    ]

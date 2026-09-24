from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from writeoff.config import Settings
from writeoff.tax_parameters import TaxParameterError, Unit, load_tax_parameters

REPO_DATA = Path(__file__).resolve().parents[1] / "data" / "tax_parameters"


def _write(tmp_path: Path, year: int, parameters: dict[str, object]) -> Path:
    (tmp_path / f"{year}.yaml").write_text(
        yaml.safe_dump({"tax_year": year, "parameters": parameters}), encoding="utf-8"
    )
    return tmp_path


def _param(**overrides: object) -> dict[str, object]:
    param: dict[str, object] = {
        "description": "Standard mileage rate",
        "unit": "usd_per_mile",
        "value": None,
        "source_url": "https://www.irs.gov/tax-professionals/standard-mileage-rates",
    }
    param.update(overrides)
    return param


@pytest.mark.parametrize("year", Settings().supported_tax_years)
def test_shipped_files_cite_sources_and_evidence(year: int) -> None:
    params = load_tax_parameters(year, REPO_DATA)
    assert params.tax_year == year
    assert "section_179_dollar_limit" in params.parameters
    for name, param in params.parameters.items():
        assert param.source_url.host is not None
        assert param.source_url.host.endswith("irs.gov"), name
        if param.is_verified:
            assert param.evidence, f"{name} is filled in without evidence"


def test_2025_values_filled_2026_still_pending() -> None:
    filled = load_tax_parameters(2025, REPO_DATA)
    assert filled.get("standard_mileage_rate_business").value == Decimal("0.70")
    assert not filled.get("qbi_threshold_single").is_verified  # not in the corpus yet
    pending = load_tax_parameters(2026, REPO_DATA)
    assert not any(p.is_verified for p in pending.parameters.values())


def test_shipped_files_mark_unverified_values() -> None:
    for path in REPO_DATA.glob("*.yaml"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("value: null"):
                assert line.rstrip().endswith("# verify on irs.gov"), f"{path.name}: {line}"


def test_filled_value_requires_evidence(tmp_path: Path) -> None:
    _write(tmp_path, 2025, {"p": _param(value="0.7")})
    with pytest.raises(TaxParameterError, match="evidence"):
        load_tax_parameters(2025, tmp_path)


def test_shipped_years_share_the_same_keys() -> None:
    keys = {
        year: set(load_tax_parameters(year, REPO_DATA).parameters)
        for year in Settings().supported_tax_years
    }
    assert len({frozenset(k) for k in keys.values()}) == 1


def test_rejects_missing_source_url(tmp_path: Path) -> None:
    param = _param()
    del param["source_url"]
    _write(tmp_path, 2025, {"standard_mileage_rate_business": param})
    with pytest.raises(TaxParameterError, match="source_url"):
        load_tax_parameters(2025, tmp_path)


@pytest.mark.parametrize("bad_url", [None, "", "not-a-url"])
def test_rejects_null_empty_or_invalid_source_url(tmp_path: Path, bad_url: object) -> None:
    _write(tmp_path, 2025, {"standard_mileage_rate_business": _param(source_url=bad_url)})
    with pytest.raises(TaxParameterError, match="source_url"):
        load_tax_parameters(2025, tmp_path)


def test_loads_verified_value_and_lookup(tmp_path: Path) -> None:
    _write(
        tmp_path, 2025, {"standard_mileage_rate_business": _param(value="0.10", evidence="test")}
    )
    params = load_tax_parameters(2025, tmp_path)
    param = params.get("standard_mileage_rate_business")
    assert param.unit is Unit.USD_PER_MILE
    assert param.is_verified
    with pytest.raises(KeyError, match="unknown tax parameter"):
        params.get("no_such_parameter")


def test_periods(tmp_path: Path) -> None:
    periods = [
        {"start": "2025-01-01", "end": "2025-06-30", "value": "1"},
        {"start": "2025-07-01", "end": "2025-12-31", "value": None},
    ]
    _write(tmp_path, 2025, {"bonus": _param(unit="percent", periods=periods, evidence="test")})
    param = load_tax_parameters(2025, tmp_path).get("bonus")
    assert param.periods is not None
    assert not param.is_verified


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "value": "1",
                "periods": [{"start": "2025-01-01", "end": "2025-12-31", "value": "1"}],
            },
            "either value or periods",
        ),
        (
            {
                "periods": [
                    {"start": "2025-01-01", "end": "2025-07-01", "value": "1"},
                    {"start": "2025-07-01", "end": "2025-12-31", "value": "2"},
                ],
                "evidence": "test",
            },
            "overlap",
        ),
        ({"unit": "furlongs"}, "unit"),
        ({"surprise": 1}, "Extra inputs"),
    ],
)
def test_rejects_invalid_parameters(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    _write(tmp_path, 2025, {"p": _param(**overrides)})
    with pytest.raises(TaxParameterError, match=message):
        load_tax_parameters(2025, tmp_path)


def test_rejects_year_mismatch(tmp_path: Path) -> None:
    _write(tmp_path, 2026, {"p": _param()})
    (tmp_path / "2026.yaml").rename(tmp_path / "2025.yaml")
    with pytest.raises(TaxParameterError, match="declares tax_year 2026"):
        load_tax_parameters(2025, tmp_path)


def test_rejects_missing_file_and_bad_yaml(tmp_path: Path) -> None:
    with pytest.raises(TaxParameterError, match="no tax parameter file"):
        load_tax_parameters(2031, tmp_path)
    (tmp_path / "2025.yaml").write_text("parameters: [unclosed", encoding="utf-8")
    with pytest.raises(TaxParameterError, match="invalid YAML"):
        load_tax_parameters(2025, tmp_path)

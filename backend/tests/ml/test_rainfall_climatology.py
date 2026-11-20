import pandas as pd

from ml.rainfall.climatology import ClimatologyParams, build_climatology


def test_one_row_per_cell_and_day_of_year(rain_daily):
    clim = build_climatology(rain_daily)
    assert len(clim) == 3 * 366
    assert not clim.duplicated(["cell_id", "doy_center"]).any()
    assert (clim["p50_mm"] <= clim["p90_mm"]).all()
    assert (clim["p90_mm"] <= clim["p99_mm"]).all()
    # ±15 days x 5 years ~ 155 observations per row
    assert clim["n_obs"].between(140, 160).all()


def test_window_wraps_around_year_end(rain_daily):
    clim = build_climatology(rain_daily).set_index(["cell_id", "doy_center"])
    assert clim.loc[(1, 1), "n_obs"] == clim.loc[(1, 180), "n_obs"]


def test_seasonal_signal_recovered():
    dates = pd.date_range("2018-01-01", "2023-12-31")
    wet_summer = dates.month.isin([6, 7, 8]) * 10.0 + 1.0
    daily = pd.DataFrame({"cell_id": 1, "obs_date": dates, "precip_mm": wet_summer})
    clim = build_climatology(daily).set_index("doy_center")
    assert clim.loc[200, "mean_mm"] == 11.0   # mid-July
    assert clim.loc[30, "mean_mm"] == 1.0     # end of January


def test_cells_with_too_few_years_are_dropped(rain_daily):
    short = rain_daily[pd.to_datetime(rain_daily["obs_date"]).dt.year >= 2022]
    assert build_climatology(short, ClimatologyParams(min_years=3)).empty

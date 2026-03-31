import numpy as np
import polars as pl
import matplotlib as plt
import datetime as dt
import sf_quant.optimizer as sfo
import sf_quant.data as sfd
import sf_quant.backtester as sfb
import os

df = pl.read_parquet( # reads in the parquet
    "alphas/alphas/alphas.parquet"
)
names = list(df['signal_name'].unique())
start_dates = [] # sort
end_dates = []
for name in names:
    factor_df = df.filter(pl.col('signal_name') == name)
    start_dates.append(factor_df.select(pl.col("date").min()).item())
    end_dates.append(factor_df.select(pl.col("date").max()).item())

start_date = max(start_dates)
end_date = min(end_dates)

columns = ['date', 'barrid', 'return', 'predicted_beta']

data = sfd.load_assets(
    start=start_date,
    end=end_date,
    in_universe=True,
    columns=columns
)

new_df = df.join(
    data,
    on=['date', 'barrid'],
    how='left'
)

combined_alpha = new_df.group_by(['date', 'barrid']).agg(pl.col('alpha').sum())

benchmark = sfd.load_benchmark(start_date, end_date)

active_constraints = [sfo.ZeroInvestment(), sfo.ZeroBeta()]

total_constraints = [sfo.FullInvestment(), sfo.LongOnly(), sfo.UnitBeta(), sfo.NoBuyingOnMargin()]

ids = list(combined_alpha['barrid'])

sfb.backtest_parallel(combined_alpha, total_constraints).write_parquet("PWeights/total.parquet")

sfb.backtest_parallel(combined_alpha, active_constraints).write_parquet("PWeights/active.parquet")
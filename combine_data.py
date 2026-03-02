import polars as pl

year_dfs = []
for i in range(1995, 2024):
    bab_df = pl.read_parquet(f"ActiveWeights/bab_weights_{i}-06-27_{i+1}-06-26.parquet").with_columns(pl.col("weight").alias('bab_weight')).drop(pl.col('weight'))
    mr_df = pl.read_parquet(f"ActiveWeights/meanrev_weights_{i}-06-27_{i+1}-06-26.parquet").with_columns(pl.col("weight").alias('mr_weight')).drop(pl.col('weight'))
    mm_df = pl.read_parquet(f"ActiveWeights/momentum_weights_{i}-06-27_{i+1}-06-26.parquet").with_columns(pl.col("weight").alias('mm_weight')).drop(pl.col('weight'))
    year_df = bab_df.join(
        mr_df, on=['date', 'barrid'], how='left'
    ).join(
        mm_df, on=['date', 'barrid'], how='left'
    )
    year_dfs.append(year_df)

df = pl.concat(year_dfs)

df.write_parquet("ActiveWeights.parquet")
import numpy as np
import polars as pl
import sf_quant.optimizer as sfo
import sf_quant.data as sfd
import datetime as dt

bab_years = list(range(1995, 2004)) + list(range(2005, 2024)) 
bab_df = pl.concat([pl.read_parquet(f"ActiveWeights/bab_weights_{i}-06-27_{i+1}-06-26.parquet") for i in bab_years]).rename({'weight': 'bab_weight'})
mr_years = list(range(1995, 2018)) + list(range(2019, 2024)) 
mr_df = pl.concat([pl.read_parquet(f"ActiveWeights/meanrev_weights_{i}-06-27_{i+1}-06-26.parquet") for i in mr_years]).rename({'weight': 'mr_weight'})
mm_df = pl.concat([pl.read_parquet(f"ActiveWeights/momentum_weights_{i}-06-27_{i+1}-06-26.parquet") for i in range(1995, 2024)]).rename({'weight': 'mm_weight'})
weights_df = bab_df.join(
    mr_df, on=['date', 'barrid'], how='left'
).join(
    mm_df, on=['date', 'barrid'], how='left'
)

def get_covs_year(start_year, filename="ActiveWeights_Missing.parquet"): # gets the portfolio covariances for the speciric year
    weights_df = pl.read_parquet(filename)
    small_weights = (
        weights_df
        .filter(
            (pl.col('date') >= dt.date(start_year, 6, 27)) &
            (pl.col('date') <= dt.date(start_year + 1, 6, 26))
        )
        .with_columns(pl.lit(0).alias('eq_alpha_weight'))
    )
    dates = (
        small_weights
        .select('date')
        .unique()
        .sort('date')
        .to_series()
        .to_list()
    )
    ids = ['bab', 'mr', 'mm']
    bab_vars = []
    mr_vars = []
    mm_vars = []
    bab_mr_cov = []
    bab_mm_cov = []
    mr_mm_cov = []

    for date in dates:
        date_df = small_weights.filter(pl.col('date') == date).sort('barrid')
        barrids = sorted([bar for bar in date_df['barrid'].unique()]) # gets the barrids
        order_map = {v: i for i, v in enumerate(barrids)} 
        date_w = date_df.with_columns( # gets the weights for the day and orders according to our barrids
            pl.col('barrid').replace(order_map).alias('order')
        ).sort('order').drop(['order', 'barrid', 'date']).to_numpy()
        cov = sfd.construct_covariance_matrix(date, barrids=barrids).drop('barrid').to_numpy() # constructs the assett covariance matrix
        new_cov = date_w.T @ cov @ date_w  # multiplies to get the portfolio covariance matrix
        bab_vars.append(new_cov[0, 0])
        mr_vars.append(new_cov[1, 1])
        mm_vars.append(new_cov[2, 2])
        bab_mr_cov.append(new_cov[0, 1])
        bab_mm_cov.append(new_cov[0, 2])
        mr_mm_cov.append(new_cov[1, 2])

    bab_vars = np.array(bab_vars)
    mr_vars = np.array(mr_vars)
    mm_vars = np.array(mm_vars)
    bab_mr_cov = np.array(bab_mr_cov)
    bab_mm_cov = np.array(bab_mm_cov)
    mr_mm_cov = np.array(mr_mm_cov)
    
    year_cov_df = pl.DataFrame({
        'date': dates,
        'bab_vars': bab_vars,
        'mr_vars': mr_vars,
        'mm_vars': mm_vars,
        'bab_mr_cov': bab_mr_cov,
        'bab_mm_cov': bab_mm_cov,
        'mr_mm_cov': mr_mm_cov
    }).sort('date')

    return year_cov_df


def check_parquets(year):
    def column_has_nonzero(col, weights_df=weights_df, year=year):
        start = dt.date(year, 6, 27)
        end   = dt.date(year + 1, 6, 26)

        return (
            weights_df
            .filter((pl.col("date") >= start) & (pl.col("date") <= end))
            .select((pl.col(col) != 0).any())
            .item()
        )
    Nonzero = True
    if not column_has_nonzero('bab_weight'):
        Nonzero = False
    if not column_has_nonzero("mr_weight"):
        Nonzero = False
    if not column_has_nonzero('mm_weight'):
        Nonzero = False
    return Nonzero

for y in range(2017, 2024):
    try: 
        if check_parquets(y):
            df = get_covs_year(y)
            df.write_parquet(f"Portfolio_covs_{y}.parquet")
    except Exception as e:
        print(e, y)
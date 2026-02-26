import numpy as np
import polars as pl
import datetime as dt
import sf_quant.data as sfd
import sf_quant.optimizer as sfo
import matplotlib.pyplot as plt

# reads in the weights parquets
t_weights_df = pl.read_parquet('weight_df.parquet')
a_weights_df = pl.read_parquet("ActiveWeights_Missing.parquet")

# gets equal sharpe-ratio priors
prior_dates = pl.date_range(
    start=dt.date(1986, 6, 26),
    end=dt.date(1996, 6, 26),
    interval='1d',
    eager=True
)
prior_n = len(prior_dates)

prior_returns = np.where(
    np.arange(prior_n) % 2,
    .045 + (np.sqrt(252) * .045),
    .045 - (np.sqrt(252) * .045)
)
prior_data = pl.DataFrame(
    {
        'date': prior_dates,
        'bab_return': prior_returns,
        'mr_return': prior_returns,
        'mm_return': prior_returns,   
    }
)

start_date = dt.date(1996, 7, 26) # gets our starting and ending date
end_date = dt.date(2004, 6, 25)

small_weights = ( # filters to our starting point
    t_weights_df
    .filter(pl.col('date') >= start_date)
    .filter(pl.col('date') <= dt.date(2004, 6, 25)
    ).with_columns(pl.lit(0).alias('eq_alpha_weight')
    ).with_columns(pl.lit(0).alias("rolling_sharpe_weight"))
)

dates = ( # Gets the dates as a series to work with
    small_weights
    .select('date')
    .unique()
    .sort('date')
    .to_series()
    .to_list()
)

data = sfd.load_assets( # loads in the data of results and joins it with the wieghts
    start=start_date,
    end=end_date,
    columns=['barrid', 'date', 'return']
)
weight_data = data.join(
    small_weights, on=['date', 'barrid'], how='left'
).sort(['barrid', 'date'])

port_returns = weight_data.with_columns( # gets the portlfio return
    pl.col('return').shift(-1).over('barrid').alias('lag_return')
).with_columns(
    (pl.col('lag_return') * pl.col('bab_weight')
    ).alias('bab_return')
).with_columns(
    (pl.col('lag_return') * pl.col('mr_weight')
    ).alias('mr_return')
).with_columns(
    (pl.col('lag_return') * pl.col('mm_weight')
    ).alias('mm_return')
).group_by('date').agg(
    pl.col('bab_return').sum(),
    pl.col('mr_return').sum(),
    pl.col('mm_return').sum()
).sort('date')

rolling_sharpe = pl.concat([prior_data, port_returns]).sort('date')
rolling_sharpe = rolling_sharpe.with_columns(
    (pl.col('bab_return').rolling_mean(window_size=2520) * np.sqrt(252) / pl.col('bab_return').rolling_std(window_size=2520)
).shift(-1).alias('bab_sharpe')
).with_columns(
    (pl.col('mr_return').rolling_mean(window_size=2520) * np.sqrt(252) / pl.col('mr_return').rolling_std(window_size=2520)
).shift(-1).alias('mr_sharpe')
).with_columns(
    (pl.col('mm_return').rolling_mean(window_size=2520) * np.sqrt(252) / pl.col('mm_return').rolling_std(window_size=2520)
).shift(-1).alias('mm_sharpe')
).with_columns(
    (pl.col('bab_sharpe') + pl.col('mr_sharpe') + pl.col('mm_sharpe')
    ).alias('sharpe_sum')
).drop_nulls(
).drop(['bab_return', 'mr_return', 'mm_return'])

# The ids are used to label the MVE
ids = ['bab', 'mr', 'mm']
# out is used to put together the results of each specific date
out = []
# these are used to track the weights of each signal to make sure we're getting a mix of each for each individual moment
eq_babs = []
eq_mrs = []
eq_mms = []
sh_babs = []
sh_mrs = []
sh_mms = []
# does the same thing but for the version with the penalty attached
pen_eq_babs = []
pen_eq_mrs = []
pen_eq_mms = []
pen_sh_babs = []
pen_sh_mrs = []
pen_sh_mms = []
pen_mix_babs = []
pen_mix_mrs = []
pen_mix_mms = []
# Adds the penalties (for equal and sharpe based, with initial values)
penalty = .6
wprev_eq = np.ones(3)/3
wprev_sh = np.ones(3)/3
wprev_mix = np.ones(3)/3

var_bab = []
cov_bab_mr = []

cov_dfs = [pl.read_parquet(f"Portfolio_covs_{i}.parquet") for i in np.arange(1996, 2004)]
cov_df = pl.concat(cov_dfs)

for date in dates[:-1]: # goes through all the dates
    if date.day == 26:
        print(date)
    date_df = small_weights.filter(pl.col('date') == date)
    bab_var, mr_var, mm_var, bab_mr_cov, bab_mm_cov, mr_mm_cov = cov_df.filter( # reads in the data from our arrays of coavariance matrices
        pl.col('date') == date
    ).row(0)[1:]
    new_cov = np.array([
        [bab_var, bab_mr_cov, bab_mm_cov],
        [bab_mr_cov, mr_var, mr_mm_cov],
        [bab_mm_cov, mr_mm_cov, mm_var]
    ])
    constraints = [sfo.FullInvestment(), sfo.LongOnly(), sfo.NoBuyingOnMargin()] # institutes our conditions
    I = np.eye(3) # gets the adj covariance matrices which will be used for the penalized version
    cov_adj = new_cov + 2*penalty * I
    eq_a = np.ones(3) / 3 # gets the equal alpha array
    eq_a_w = sfo.mve_optimizer( # runs MVE on this
        ids=ids,
        alphas=eq_a,
        covariance_matrix=new_cov,
        constraints=constraints
    )

    eq_w = np.array([d["weight"] for d in eq_a_w.to_dicts()]) # appends the weight for each signal portfolio to the lists
    eq_babs.append(eq_w[0])
    eq_mrs.append(eq_w[1])
    eq_mms.append(eq_w[2])

    pen_eq_a_w = sfo.mve_optimizer( # does the same thing but with the modified version
        ids=ids,
        alphas=eq_a + 2*penalty*wprev_eq,
        covariance_matrix=cov_adj,
        constraints=constraints
    )
    wprev_eq = np.array([d["weight"] for d in pen_eq_a_w.to_dicts()]) 
    pen_eq_babs.append(wprev_eq[0])
    pen_eq_mrs.append(wprev_eq[1])
    pen_eq_mms.append(wprev_eq[2])

    bab_sharpe, mr_sharpe, mm_sharpe, sharpe_sum = rolling_sharpe.filter( # reads in the data our rolling sharpe ratios
        pl.col('date') == date
    ).row(0)[1:]

    sharpe_a = np.array([bab_sharpe, mr_sharpe, mm_sharpe]) # does the same thing except with the sharpe-based alphas
    sh_a_w = sfo.mve_optimizer(
        ids=ids,
        alphas=sharpe_a,
        covariance_matrix=new_cov,
        constraints=constraints
    )
    sh_w = np.array([d["weight"] for d in sh_a_w.to_dicts()]) 
    sh_babs.append(sh_w[0])
    sh_mrs.append(sh_w[1])
    sh_mms.append(sh_w[2])

    pen_sh_a_w = sfo.mve_optimizer( # does the same thing but with the modified version
        ids=ids,
        alphas=sharpe_a + 2*penalty*wprev_sh,
        covariance_matrix=cov_adj,
        constraints=constraints
    )
    wprev_sh = np.array([d["weight"] for d in pen_sh_a_w.to_dicts()]) 
    pen_sh_babs.append(wprev_sh[0])
    pen_sh_mrs.append(wprev_sh[1])
    pen_sh_mms.append(wprev_sh[2])

    pen_mix_w = sfo.mve_optimizer( # does the same thing but with the modified version
        ids=ids,
        alphas=(.3 * sharpe_a) + (.7 * eq_a) + 2*penalty*wprev_mix,
        covariance_matrix=cov_adj,
        constraints=constraints
    )
    wprev_mix = np.array([d["weight"] for d in pen_mix_w.to_dicts()]) 
    pen_mix_babs.append(wprev_mix[0])
    pen_mix_mrs.append(wprev_mix[1])
    pen_mix_mms.append(wprev_mix[2])

    lam_eqa = { # gets lambdas to calculate the finals weights based on the 2 strategies
    row['barrid']: row['weight']
    for row in eq_a_w.to_dicts()
    }

    lam_sha = {
    row['barrid']: row['weight']
    for row in sh_a_w.to_dicts()
    }

    lam_peqa = {
    row['barrid']: row['weight']
    for row in pen_eq_a_w.to_dicts()
    }

    lam_psha = {
    row['barrid']: row['weight']
    for row in pen_sh_a_w.to_dicts()
    }

    lam_pmix = {
    row['barrid']: row['weight']
    for row in pen_mix_w.to_dicts()
    }

    a_date_df = a_weights_df.filter(pl.col('date') == date)

    a_date_df = a_date_df.with_columns(
        (pl.col('bab_weight') * lam_eqa['bab'] +
        pl.col('mm_weight') * lam_eqa['mm'] +
        pl.col('mr_weight') * lam_eqa['mr']
        ).alias('eqa_weight')
    ).with_columns(
        (pl.col('bab_weight') * lam_sha['bab'] +
        pl.col('mm_weight') * lam_sha['mm'] +
        pl.col('mr_weight') * lam_sha['mr']
        ).alias('sha_weight')
    ).with_columns(
        (pl.col('bab_weight') * lam_peqa['bab'] +
        pl.col('mm_weight') * lam_peqa['mm'] +
        pl.col('mr_weight') * lam_peqa['mr']
        ).alias('pen_eqa_weight')
    ).with_columns(
        (pl.col('bab_weight') * lam_psha['bab'] +
        pl.col('mm_weight') * lam_psha['mm'] +
        pl.col('mr_weight') * lam_psha['mr']
        ).alias('pen_sha_weight')
    ).with_columns(
        (pl.col('bab_weight') * lam_pmix['bab'] +
        pl.col('mm_weight') * lam_pmix['mm'] +
        pl.col('mr_weight') * lam_pmix['mr']
        ).alias('pen_mix_weight')
    ).with_columns(
       ((pl.col('bab_weight') / 3) +
        (pl.col('mm_weight') / 3) +
        (pl.col('mr_weight') / 3)
        ).alias('eq_weight') 
    ).with_columns(
       ((pl.col('bab_weight') * (bab_sharpe / sharpe_sum)) +
        (pl.col('mm_weight') * (mm_sharpe / sharpe_sum)) +
        (pl.col('mr_weight') * (mr_sharpe / sharpe_sum))
        ).alias('sh_weight') 
    ).with_columns(
       ((pl.col('bab_weight') * ((.3 + bab_sharpe) / (.9 + sharpe_sum))) +
        (pl.col('mm_weight') * ((.3 + mm_sharpe) / (.9 + sharpe_sum))) +
        (pl.col('mr_weight') * ((.3 + mr_sharpe) / (.9 + sharpe_sum)))
        ).alias('m_sh_weight') 
    )
    out.append(a_date_df)

total_weights = pl.concat(out) # creates total weights by concatenating the list of dailys in out

x = np.linspace(0, 1, len(eq_babs)) # plots the different wieghts 
plt.plot(x, eq_babs, label='Equal Alpha BAB Weights')
plt.plot(x, eq_mrs, label='Equal Alpha MR Weights')
plt.plot(x, eq_mms, label='Equal Alpha Momentum Weights')
plt.legend()
plt.savefig("EQ_Alpha_Weights.png", dpi=300, bbox_inches="tight")
plt.close()
plt.plot(x, sh_babs, label='Sharpe Alpha BAB Weights')
plt.plot(x, sh_mrs, label='Sharpe Alpha MR Weights')
plt.plot(x, sh_mms, label='Sharpe Alpha Momentum Weights')
plt.legend()
plt.savefig("SH_Alpha_Weights.png", dpi=300, bbox_inches="tight")
plt.close()
plt.plot(x, pen_eq_babs, label='Penalized Equal Alpha BAB Weights')
plt.plot(x, pen_eq_mrs, label='Penalized Equal Alpha MR Weights')
plt.plot(x, pen_eq_mms, label='Penalized Equal Alpha Momentum Weights')
plt.legend()
plt.savefig("Pen_EQ_Alpha_Weights.png", dpi=300, bbox_inches="tight")
plt.close()
plt.plot(x, pen_sh_babs, label='Penalized Sharpe Alpha BAB Weights')
plt.plot(x, pen_sh_mrs, label='Penalized Sharpe Alpha MR Weights')
plt.plot(x, pen_sh_mms, label='Penalized Sharpe Alpha Momentum Weights')
plt.legend()
plt.savefig("Pen_SH_Alpha_Weights.png", dpi=300, bbox_inches="tight")
plt.close()
plt.plot(x, pen_mix_babs, label='Penalized Mixed Alpha BAB Weights')
plt.plot(x, pen_mix_mrs, label='Penalized Mixed Alpha MR Weights')
plt.plot(x, pen_mix_mms, label='Penalized Mixed Alpha Momentum Weights')
plt.legend()
plt.savefig("Pen_Mix_Alpha_Weights.png", dpi=300, bbox_inches="tight")
plt.close()


# Gets the data
data = sfd.load_assets(
    start=dates[0] - dt.timedelta(days=3),
    end=dates[-1] + dt.timedelta(days=3),
    columns=['barrid', 'date', 'return']
)
weight_data = data.join(
    total_weights, on=['date', 'barrid'], how='left'
)

weight_data = weight_data.sort(['barrid', 'date'])

weight_data = weight_data.with_columns(
    pl.col('return').shift(-1).over('barrid')
    .alias('lag_return')
).with_columns(
    (pl.col('eqa_weight') * pl.col('lag_return'))
    .alias('eqa_return')
).with_columns(
    (pl.col('sh_weight') * pl.col('lag_return'))
    .alias('sh_return')
).with_columns(
    (pl.col('m_sh_weight') * pl.col('lag_return'))
    .alias('m_sh_return')
).with_columns(
    (pl.col('eq_weight') * pl.col('lag_return'))
    .alias('eq_return')
).with_columns(
    (pl.col('sha_weight') * pl.col('lag_return'))
    .alias('sha_return')
).with_columns(
    (pl.col('pen_eqa_weight') * pl.col('lag_return'))
    .alias('pen_eqa_return')
).with_columns(
    (pl.col('pen_sha_weight') * pl.col('lag_return'))
    .alias('pen_sha_return')
).with_columns(
    (pl.col('pen_mix_weight') * pl.col('lag_return'))
    .alias('pen_mix_return')
).drop_nulls()

results = weight_data.group_by('date').agg(
    [
        pl.col('eqa_return').sum().alias('eqa_ret'),
        pl.col('eq_return').sum().alias('eq_ret'),
        pl.col('sh_return').sum().alias('sh_ret'),
        pl.col('m_sh_return').sum().alias('m_sh_ret'),
        pl.col('sha_return').sum().alias('sha_ret'),
        pl.col('pen_eqa_return').sum().alias('pen_eqa_ret'),
        pl.col('pen_sha_return').sum().alias('pen_sha_ret'),
        pl.col('pen_mix_return').sum().alias('pen_mix_ret')
    ]
).with_columns(
    pl.col('eqa_ret').truediv(100).log1p()
    .alias('log_eqa_ret')
).with_columns(
    pl.col('sh_ret').truediv(100).log1p()
    .alias('log_sh_ret')
).with_columns(
    pl.col('m_sh_ret').truediv(100).log1p()
    .alias('log_m_sh_ret')
).with_columns(
    pl.col('eq_ret').truediv(100).log1p()
    .alias('log_eq_ret')
).with_columns(
    pl.col('sha_ret').truediv(100).log1p()
    .alias('log_sha_ret')
).with_columns(
    pl.col('pen_eqa_ret').truediv(100).log1p()
    .alias('log_pen_eqa_ret')
).with_columns(
    pl.col('pen_sha_ret').truediv(100).log1p()
    .alias('log_pen_sha_ret')
).with_columns(
    pl.col('pen_mix_ret').truediv(100).log1p()
    .alias('log_pen_mix_ret')
).sort('date'
).with_columns(
    pl.col('log_eqa_ret').cum_sum()
    .alias('c_log_eqa_ret')
).with_columns(
    pl.col('log_eq_ret').cum_sum()
    .alias('c_log_eq_ret')
).with_columns(
    pl.col('log_sh_ret').cum_sum()
    .alias('c_log_sh_ret')
).with_columns(
    pl.col('log_m_sh_ret').cum_sum()
    .alias('c_log_m_sh_ret')
).with_columns(
    pl.col('log_sha_ret').cum_sum()
    .alias('c_log_sha_ret')
).with_columns(
    pl.col('log_pen_sha_ret').cum_sum()
    .alias('c_log_pen_sha_ret')
).with_columns(
    pl.col('log_pen_eqa_ret').cum_sum()
    .alias('c_log_pen_eqa_ret')
).with_columns(
    pl.col('log_pen_mix_ret').cum_sum()
    .alias('c_log_pen_mix_ret')
)
results

eq_w_sharpe = (results['eq_ret'].mean() / results['eq_ret'].std()) * np.sqrt(252)
eqa_sharpe = (results['eqa_ret'].mean() / results['eqa_ret'].std()) * np.sqrt(252)
sha_sharpe = (results['sha_ret'].mean() / results['sha_ret'].std()) * np.sqrt(252)
sh_sharpe = (results['sh_ret'].mean() / results['sh_ret'].std()) * np.sqrt(252)
m_sh_sharpe = (results['m_sh_ret'].mean() / results['m_sh_ret'].std()) * np.sqrt(252)
pen_eqa_sharpe = (results['pen_eqa_ret'].mean() / results['pen_eqa_ret'].std()) * np.sqrt(252)
pen_sha_sharpe = (results['pen_sha_ret'].mean() / results['pen_sha_ret'].std()) * np.sqrt(252)
pen_mix_sharpe = (results['pen_mix_ret'].mean() / results['pen_mix_ret'].std()) * np.sqrt(252)


plt.plot(results['date'], results['c_log_eq_ret'], label=f'Equal Weight, Sharpe={round(eq_w_sharpe, 3)}')
plt.plot(results['date'], results['c_log_eqa_ret'], label=f'Equal Alphas MVE, Sharpe={round(eqa_sharpe, 3)}')
plt.plot(results['date'], results['c_log_sha_ret'], label=f'Sharpe Alpha MVE, Sharpe={round(sha_sharpe, 3)}')
plt.plot(results['date'], results['c_log_sh_ret'], label=f'Sharpe Weight, Sharpe={round(sh_sharpe, 3)}')
plt.plot(results['date'], results['c_log_m_sh_ret'], label=f'Mixed Sharpe Weight, Sharpe={round(m_sh_sharpe, 3)}')
plt.plot(results['date'], results['c_log_pen_sha_ret'], label=f'Penalzied Sharpe Alpha MVE, Sharpe={round(pen_sha_sharpe, 3)}')
plt.plot(results['date'], results['c_log_pen_eqa_ret'], label=f'Penalized Equal Alpha MVE, Sharpe={round(pen_eqa_sharpe, 3)}')
plt.plot(results['date'], results['c_log_pen_mix_ret'], label=f'Penalized Mixed Alpha MVE, Sharpe={round(pen_mix_sharpe, 3)}')



plt.legend()
plt.xlabel('date')
plt.ylabel('Cumulative Log Return')
plt.savefig("A_Results.png")

with open("a_mixed_sharpes.txt", "w") as f:
    f.write(f"eq_w_sharpe: {eq_w_sharpe}, eqa_sharpe: {eqa_sharpe}, sh_sharpe: {sh_sharpe}, m_sh_sharpe: {m_sh_sharpe}, sha_sharpe: {sha_sharpe}, pen_eqa_sharpe: {pen_eqa_sharpe}, pen_sha_sharpe: {pen_sha_sharpe}, pen_mix_sharpe: {pen_mix_sharpe}")
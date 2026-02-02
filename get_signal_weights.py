import sf_quant.optimizer as sfo
import sf_quant.backtester as sfb
import polars as pl
import datetime as dt
import argparse
import os

def get_signal_weights(df:pl.LazyFrame, signal: str, start, end, n_cpus=8, write=False, write_path=None, gamma=2):
    """
    Get signal for df of date, barrid, and signal alpha.

    Parameters
    ----------
    data : pl.DataFrame
        Input dataset containing at least the following columns:

        - ``date`` : datetime-like, the date of each observation.
        - ``barrid`` : str, unique identifier for each asset.
        - ``f{signal}_alpha : float, signal alpha for each barrid for each date

    signal : str, signal name
    start : datetime-like, the start date
    end : datetime-like, the end date
    n_cpus : int, optional
        Number of CPUs to allocate to Ray. Defaults to
        8 but is capped at the number of unique dates.
    gamma : float, optional
        Risk aversion parameter. Higher values penalize portfolio
        variance more strongly. Default is 2.
    write : Bool, whether to write, defaults to False
    write_path : str, write path. 
        Default is None, which causes write to f'weights/{signal}_weights_{start}_{end}.parquet'
        if write is True

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing optimized portfolio weights across
        all backtest dates, with columns:

        - ``date`` : datetime, portfolio date.
        - ``barrid`` : str, asset identifier.
        - ``weight`` : float, optimized portfolio weight for signal alphas
    
    """
    # Lazy frames make several operations slightly faster
    filtered = (
        df.filter(
        (pl.col('date') >= start) &
        (pl.col('date') <= end) &
        (pl.col(f"{signal}_alpha").is_not_null())
        )
        .select(['date', 'barrid', f'{signal}_alpha', 'predicted_beta'
        ])
        .collect()
    )

    # This frequently happens for the first time period of historical signals, e.g. momentum
    if filtered.is_empty(): 
        print("[WARNING] After filtering, input df was empty.")
        return None

    constraints = [
        sfo.FullInvestment(),
        sfo.LongOnly(),
        sfo.NoBuyingOnMargin(),
        sfo.UnitBeta()
    ]

    weights = sfb.backtest_parallel(filtered.rename({f'{signal}_alpha': 'alpha'}), constraints, gamma=gamma, n_cpus=n_cpus)

    # Check nothing terrible has happened before writing

    if weights.is_empty():
        print(f"[WARNING] {signal} {start}–{end}: weights output is EMPTY")
    
    # Check sums are as expected
    else:
        n_dates = weights.select(pl.col("date")).n_unique()
        total_weight = weights.select(pl.col("weight")).sum().item()
        print(f"[INFO] {signal} {start}–{end}: {n_dates} dates, total weight sum = {total_weight:.6f}")
    
    # Write to parquet
    if write: 
        if not write_path:
            weights.write_parquet(f'weights/{signal}_weights_{start}_{end}.parquet')
        else:
            weights.write_parquet(write_path)

    return weights

if __name__ == '__main__':
    # These prints here help debug, prob should be a debug mode lol

    parser = argparse.ArgumentParser(
        description="Run signal weighting on a parquet dataset."
    )
    # Parser args
    parser.add_argument("parquet", help="Path to parquet file containing the data")
    parser.add_argument("signal", help="Signal name (without _alpha suffix)")
    parser.add_argument("start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the output parquet with weights to disk",
    )

    args = parser.parse_args()

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    print(f"Ray: Slurm allocated {n_cpus} CPUs...")

    # Parse dates into datetime.date objects
    start = dt.date.fromisoformat(args.start)
    print(f"Starting at {start}...")
    end = dt.date.fromisoformat(args.end)
    print(f"Ending at {end}...")

    # Load parquet into polars DataFrame
    print(f"Loading data from {args.parquet}...")
    df = pl.scan_parquet(args.parquet)
    
    # This should be replaced with (monthly?) gamma tuning, I was just guessing
    gamma = 2
    if args.signal == 'momentum':
        gamma = 10

    if args.signal == 'meanrev':
        gamma = 20

    if args.signal == 'bab':
        gamma = 4

    print(f'Set gamma to {gamma}...')

    # Run the signal weights calculation
    print(f"Starting MVO...")
    weights = get_signal_weights(df, args.signal, start, end, n_cpus=min(8, n_cpus), write=args.write, gamma=gamma)
    print("Done!")
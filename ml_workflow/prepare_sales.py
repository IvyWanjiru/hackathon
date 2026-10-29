"""UCI practice adapter: gross positive completed sales per SKU and Monday week.

This target excludes returns/cancellations and is NOT net revenue or unmet demand.
The original file does not establish active listing/stock status in missing weeks.
Review that assumption before using its generated zero-filled practice series.
"""
import argparse
from pathlib import Path

from .common import digest, write_json


def prepare(path, output, assume_active=False, max_skus=20):
    import pandas as pd
    if not assume_active:
        raise ValueError("For this practice dataset, explicitly pass --assume-active-between-first-last after reading DATASETS.md.")
    sheets = pd.read_excel(path, sheet_name=None)
    frames = []
    for frame in sheets.values():
        frame = frame.rename(columns={"Invoice": "InvoiceNo", "Price": "UnitPrice"})
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    required = {"InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice"}
    if not required <= set(raw):
        raise ValueError("Expected UCI Online Retail II workbook columns.")
    raw_count = len(raw)
    raw["InvoiceDate"] = pd.to_datetime(raw.InvoiceDate, errors="coerce")
    raw["Quantity"] = pd.to_numeric(raw.Quantity, errors="coerce")
    raw["UnitPrice"] = pd.to_numeric(raw.UnitPrice, errors="coerce")
    cancelled = raw.InvoiceNo.astype(str).str.upper().str.startswith("C")
    clean = raw[~cancelled & raw.StockCode.notna() & raw.InvoiceDate.notna() & raw.Quantity.gt(0) & raw.UnitPrice.gt(0)].copy()
    # The workbook lacks invoice-line IDs. Keep repeated lines: do not assume
    # two identical-looking lines are accidental duplicates without evidence.
    clean["week"] = clean.InvoiceDate.dt.to_period("W-SUN").dt.start_time
    # Drop both boundary weeks; they may have only partial coverage.
    clean = clean[(clean.week > clean.week.min()) & (clean.week < clean.week.max())]
    clean["sku"] = clean.StockCode.astype(str)
    weekly = clean.groupby(["sku", "week"], as_index=False).Quantity.sum().rename(columns={"Quantity": "quantity"})
    # Choose SKUs using only the early history, not future test-period volume.
    selection_end = weekly.week.min() + pd.Timedelta(weeks=12)
    first_period = weekly[weekly.week < selection_end]
    chosen = first_period.groupby("sku").quantity.sum().sort_values(ascending=False).head(max_skus).index
    series = []
    for sku, group in weekly[weekly.sku.isin(chosen)].groupby("sku"):
        calendar = pd.date_range(group.week.min(), group.week.max(), freq="W-MON")
        if len(calendar) < 67:
            continue
        filled = group.set_index("week").quantity.reindex(calendar, fill_value=0)
        series.extend(dict(sku=sku, week=date.date().isoformat(), quantity=float(value)) for date, value in filled.items())
    if not series:
        raise ValueError("No sufficiently long SKU series. Inspect the workbook/date coverage.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(series).to_csv(output/"weekly_sales.csv", index=False)
    write_json(output/"manifest.json", dict(source="https://archive.ics.uci.edu/dataset/502/online+retail+ii", license="CC BY 4.0",
               input_sha256=digest(path), raw_rows=raw_count, retained_positive_rows=len(clean),
               cancellation_rows=int(cancelled.sum()), selected_skus=len({row['sku'] for row in series}), weekly_rows=len(series),
               target="Gross positive quantity; cancellations/returns excluded; boundary weeks excluded.",
               warning="Zero fill assumes each chosen SKU was actively offered continuously between first and last observed sale. UCI does not verify this. Practice only; use listing and availability history in a retailer pilot."))
    print(f"Wrote {len(series)} practice SKU-weeks. Read the zero-fill assumption in manifest.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", required=True)
    parser.add_argument("--out", default="ml_workflow/work/uci")
    parser.add_argument("--max-skus", type=int, default=20)
    parser.add_argument("--assume-active-between-first-last", action="store_true")
    args = parser.parse_args()
    if args.max_skus < 1:
        parser.error("--max-skus must be positive")
    prepare(args.xlsx, args.out, args.assume_active_between_first_last, args.max_skus)

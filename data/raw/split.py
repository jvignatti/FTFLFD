import pandas as pd
from pathlib import Path

input_file = "vt_crashes_all.csv"
output_dir = Path("yearly")
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(input_file)

dates = pd.to_datetime(df["ACCIDENTDATE"], unit="ms", errors="coerce")

for year in range(2010, 2027):
    year_df = df[dates.dt.year == year]
    year_df.to_csv(output_dir / f"vt_crashes_{year}.csv", index=False)

print("Done. Created files vt_crashes_2010.csv through vt_crashes_2026.csv")
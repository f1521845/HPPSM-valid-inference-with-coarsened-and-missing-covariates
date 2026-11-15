"""Build a per-stay summary table from the PhysioNet/CinC 2012 raw files (sets A, B, C).
For each time-series variable: mean (oracle 48-h level), min, max, first value, count.
Values of -1 (missing) and physiologically impossible zeros are dropped."""
import sys, os, glob
import numpy as np, pandas as pd
src = sys.argv[1]; out = sys.argv[2]
frames = []
for s in ["set-a", "set-b", "set-c"]:
    for f in glob.glob(os.path.join(src, s, "*.txt")):
        d = pd.read_csv(f)
        rid = int(d.loc[d.Parameter == "RecordID", "Value"].iloc[0])
        d["RecordID"] = rid
        frames.append(d)
L = pd.concat(frames, ignore_index=True)
L = L[L.Parameter != "RecordID"]
L["Value"] = pd.to_numeric(L.Value, errors="coerce")
L["hours"] = L.Time.str.split(":").apply(lambda x: int(x[0]) + int(x[1]) / 60)
desc = ["Age", "Gender", "Height", "ICUType", "Weight"]
D = L[L.Parameter.isin(desc) & (L.hours == 0)].pivot_table(index="RecordID", columns="Parameter", values="Value", aggfunc="first")
T = L[~L.Parameter.isin(desc) | (L.hours > 0)].copy()
T = T[T.Value >= 0]
# zeros are impossible for these physiological variables
nonzero = ["HR", "MAP", "NIMAP", "Temp", "BUN", "Creatinine", "HCT", "WBC", "Platelets", "Glucose", "Na", "K", "HCO3", "GCS",
           "RespRate", "pH", "PaO2", "PaCO2", "Albumin", "Bilirubin", "Lactate", "FiO2", "Mg"]
T = T[~(T.Parameter.isin(nonzero) & (T.Value <= 0))]
T = T.sort_values(["RecordID", "Parameter", "hours"])
g = T.groupby(["RecordID", "Parameter"]).Value
S = pd.concat({"mean": g.mean(), "min": g.min(), "max": g.max(), "first": g.first(), "n": g.size()}, axis=1).unstack("Parameter")
S.columns = [f"{p}_{stat}" for stat, p in S.columns]
# urine output: total over 48 h (sum), mechanical ventilation: any record
u = T[T.Parameter == "Urine"].groupby("RecordID").Value.sum().rename("Urine_total")
mv = T[T.Parameter == "MechVent"].groupby("RecordID").Value.max().rename("MechVent_any")
O = pd.concat([pd.read_csv(os.path.join(src, f"Outcomes-{k}.txt")) for k in "abc"]).set_index("RecordID")
X = D.join(S, how="left").join(u).join(mv).join(O)
X["MechVent_any"] = X.MechVent_any.fillna(0)
X.to_pickle(out)
print(X.shape, "in-hospital deaths:", int(X["In-hospital_death"].sum()))

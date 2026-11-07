from sklearn.cluster import KMeans

ENGINE, CYCLE = "Engine", "Cycles"
OP_COLS = ["OP_Setting 1", "OP_Setting 2", "OP_Setting 3"]
# Sensors that carry (almost) no degradation signal; adjust if your column
# numbering differs. Set to [] to keep everything.
DROP_SENSOR_IDS = [1, 5, 6, 10, 16, 18, 19]

WINDOW = 30
RUL_CAP = 125
N_CONDITIONS = 6
VAL_FRACTION = 0.2
N_NN_SEEDS = 3          # size of the CNN-GRU ensemble
EPOCHS = 20
BATCH_SIZE = 128
SEED = 42

def get_sensor_cols(df):
    drop = {f"Sensor_{i}" for i in DROP_SENSOR_IDS}
    return [c for c in df.columns if c.startswith("Sensor_") and c not in drop]

def fit_preprocessor(train_df, sensors):
    """Fit on TRAIN engines only. Returns a dict that can be joblib-dumped."""
    km = KMeans(n_clusters=N_CONDITIONS, n_init=10, random_state=SEED)
    km.fit(train_df[OP_COLS])
    cond = km.predict(train_df[OP_COLS])
    grp = train_df[sensors].groupby(cond)
    mean = grp.mean()
    std = grp.std().replace(0, 1.0).fillna(1.0)
    return { "km": km, "sensors": sensors, "mean": mean, "std": std}


def apply_preprocessor(df, pre):
    df = df.copy()
    cond = pre["km"].predict(df[OP_COLS])
    mu = pre["mean"].loc[cond].values
    sd = pre["std"].loc[cond].values
    df[pre["sensors"]] = (df[pre["sensors"]].values - mu) / sd
    return df 
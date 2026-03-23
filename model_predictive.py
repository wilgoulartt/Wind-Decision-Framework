# train_hourly.py
# ==========================================================
# PREVISÃO DE VENTO (ERA5 HOURLY) — 17 TURBINAS
# Modelo: TCN multi-horizonte 
# Inclui: Ablation Study + Wilcoxon + Diebold–Mariano (HAC)
# ==========================================================
# Espera um CSV com colunas (mínimo):
#   id, datetime, u10, v10, elevation, slope
# (se wind_speed_80m não existir, será calculado a partir de u10/v10 e log-law)
#
# Saídas:
#   - métricas por horizonte (t+3, t+6, t+12) vs persistência
#   - ablation comparando variantes TCN
#   - testes estatísticos (Wilcoxon pointwise / por turbina, DM-HAC) para o melhor modelo vs persistência
#
# Execução:
#   python train_hourly.py --csv /caminho/ERA5_HOURLY_17_TURBINAS_2023_2024.csv
# ==========================================================

import argparse
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Stats
from scipy.stats import wilcoxon
from scipy.stats import norm


# -------------------------
# Config padrão
# -------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

LOOKBACK = 24
H_MAX = 12
H_EVAL = [3, 6, 12]

BATCH_SIZE = 256
EPOCHS = 30

# Split temporal (igual ao seu)
TRAIN_END = "2024-01-01"
VAL_END   = "2024-07-01"  # [TRAIN_END, VAL_END) = validação; >= VAL_END = teste


# ==========================================================
# Utils
# ==========================================================
def ajustar_velocidade_altura(v_ref, z_ref=10, z_dest=80, z0=0.1):
    return v_ref * (np.log(z_dest / z0) / np.log(z_ref / z0))


def mae_rmse(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse


def split_masks_hourly(df):
    dt = df["datetime"]
    train_mask = dt < pd.Timestamp(TRAIN_END)
    val_mask   = (dt >= pd.Timestamp(TRAIN_END)) & (dt < pd.Timestamp(VAL_END))
    test_mask  = dt >= pd.Timestamp(VAL_END)
    return train_mask, val_mask, test_mask


def add_time_features(df):
    dt = df["datetime"]
    hour = dt.dt.hour
    doy = dt.dt.dayofyear

    df["hour_sin"] = np.sin(2*np.pi*hour/24)
    df["hour_cos"] = np.cos(2*np.pi*hour/24)
    df["doy_sin"]  = np.sin(2*np.pi*doy/365.25)
    df["doy_cos"]  = np.cos(2*np.pi*doy/365.25)
    return df


def fit_scalers(train_df, features, target):
    mu = train_df[features].mean()
    sd = train_df[features].std().replace(0, 1.0)

    mu_y = train_df[target].mean()
    sd_y = train_df[target].std()
    if sd_y == 0 or np.isnan(sd_y):
        sd_y = 1.0
    return mu, sd, float(mu_y), float(sd_y)


def apply_scalers(df, mu, sd, mu_y, sd_y, features, target):
    X = ((df[features] - mu) / sd).values.astype(np.float32)
    y = ((df[target] - mu_y) / sd_y).values.astype(np.float32)
    return X, y


# ==========================================================
# Dataset multi-horizonte (NÃO Seq2Seq)
# X: (N, LOOKBACK, F)
# Y: (N, H_MAX) -> y(t+1..t+Hmax) normalizado
# y_t_real: (N,) -> y(t) em real (persistência)
# t0_dt: (N,) -> datetime do tempo t
# ==========================================================
def make_multiH_sequences_per_id(df_split, lookback, h_max,
                                features, target,
                                mu, sd, mu_y, sd_y):
    X_list, Y_list = [], []
    id_list, t0_list = [], []
    y_t_real_list = []

    for tid, g in df_split.groupby("id"):
        g = g.sort_values("datetime").reset_index(drop=True)
        if len(g) < lookback + h_max + 1:
            continue

        Xg, yg = apply_scalers(g, mu, sd, mu_y, sd_y, features, target)

        # janela termina em i (tempo t), prevê i+1..i+H
        for i in range(lookback - 1, len(g) - h_max):
            X_seq = Xg[i - (lookback - 1): i + 1, :]         # (lookback, F)
            Y_seq = yg[i + 1: i + h_max + 1]                 # (h_max,)
            t0 = g.loc[i, "datetime"]
            y_t_real = float(g.loc[i, target])               # persistência em real

            X_list.append(X_seq)
            Y_list.append(Y_seq)
            id_list.append(int(tid))
            t0_list.append(t0)
            y_t_real_list.append(y_t_real)

    if len(X_list) == 0:
        return None

    X_out = np.array(X_list, dtype=np.float32)
    Y_out = np.array(Y_list, dtype=np.float32)
    id_out = np.array(id_list, dtype=int)
    t0_out = pd.to_datetime(np.array(t0_list))
    y_t_real_out = np.array(y_t_real_list, dtype=np.float32)

    return X_out, Y_out, t0_out, id_out, y_t_real_out


# ==========================================================
# TCN
# ==========================================================
def tcn_residual_block(x, filters, kernel_size, dilation_rate, dropout=0.15):
    prev = x

    x = layers.Conv1D(filters, kernel_size,
                      padding="causal",
                      dilation_rate=dilation_rate,
                      activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv1D(filters, kernel_size,
                      padding="causal",
                      dilation_rate=dilation_rate,
                      activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    if prev.shape[-1] != filters:
        prev = layers.Conv1D(filters, 1, padding="same")(prev)

    x = layers.Add()([prev, x])
    x = layers.LayerNormalization()(x)
    return x


def build_tcn_model(lookback, n_features, h_max,
                    filters=64, kernel_size=3,
                    dilations=(1, 2, 4, 8, 16),
                    dropout=0.15, lr=1e-3):
    inp = layers.Input(shape=(lookback, n_features))
    x = inp
    for d in dilations:
        x = tcn_residual_block(x, filters, kernel_size, d, dropout)

    x = layers.Lambda(lambda z: z[:, -1, :])(x)  # representação no tempo t
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(h_max)(x)                 # multi-horizonte (t+1..t+Hmax)

    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return model


# ==========================================================
# Avaliação por horizonte (modelo vs persistência)
# ==========================================================
def denorm(y_n, mu_y, sd_y):
    return (y_n * sd_y) + mu_y


def evaluate_horizons(Y_real, P_real, y_t_real, H_eval=H_EVAL):
    rows = []
    for H in H_eval:
        h = H - 1
        y_true = Y_real[:, h]
        y_pred = P_real[:, h]
        y_base = y_t_real  # persistência v(t)

        mae_m, rmse_m = mae_rmse(y_true, y_pred)
        mae_b, rmse_b = mae_rmse(y_true, y_base)

        rows.append({
            "H_horas": H,
            "MODEL_MAE_TEST": mae_m,
            "MODEL_RMSE_TEST": rmse_m,
            "BASE_MAE_TEST": mae_b,
            "BASE_RMSE_TEST": rmse_b,
            "GAIN_MAE": mae_b - mae_m,
            "GAIN_RMSE": rmse_b - rmse_m
        })
    return pd.DataFrame(rows)


# ==========================================================
# Stats: Wilcoxon + DM (HAC / Newey-West)
# ==========================================================
def make_test_df_for_stats(id_te, t0_te, Yte_n, Pte_n, yte_t_real, mu_y, sd_y, H):
    """
    Monta DF com erros pareados no TEST para um horizonte H:
      y_true(t+H), y_pred(t+H), y_base(t) (persistência)
    """
    h = H - 1
    dt_target = t0_te + pd.to_timedelta(H, unit="h")

    y_true = denorm(Yte_n[:, h], mu_y, sd_y)
    y_pred = denorm(Pte_n[:, h], mu_y, sd_y)
    y_base = yte_t_real

    df = pd.DataFrame({
        "id": id_te,
        "datetime": dt_target,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_base": y_base
    }).sort_values(["id", "datetime"]).reset_index(drop=True)
    return df


def wilcoxon_pointwise(df):
    e_model = np.abs(df["y_true"] - df["y_pred"]).values
    e_base  = np.abs(df["y_true"] - df["y_base"]).values
    d = e_base - e_model  # >0 => modelo melhor

    # Wilcoxon exige diferenças não todas zero
    if np.allclose(d, 0):
        return {"wilcoxon_stat": np.nan, "p_value": 1.0, "median_gain": 0.0}

    stat, p = wilcoxon(d)
    return {"wilcoxon_stat": float(stat), "p_value": float(p), "median_gain": float(np.median(d))}


def wilcoxon_by_turbine(df):
    g = df.groupby("id").apply(lambda x: pd.Series({
        "mae_model": np.mean(np.abs(x["y_true"] - x["y_pred"])),
        "mae_base":  np.mean(np.abs(x["y_true"] - x["y_base"])),
    })).reset_index()

    d = (g["mae_base"] - g["mae_model"]).values
    if np.allclose(d, 0):
        return g, {"wilcoxon_stat": np.nan, "p_value": 1.0, "median_gain": 0.0}

    stat, p = wilcoxon(d)
    return g, {"wilcoxon_stat": float(stat), "p_value": float(p), "median_gain": float(np.median(d))}


def dm_test_hac(d_t, h=1):
    """
    Diebold–Mariano (aprox normal) com variância HAC/Newey-West.
    d_t = L_base(t) - L_model(t) (positivo => modelo melhor)
    """
    d_t = np.asarray(d_t, dtype=float)
    d_t = d_t[~np.isnan(d_t)]
    T = len(d_t)
    if T < 30:
        return {"DM": np.nan, "p_value": np.nan, "T": int(T), "mean_gain": float(np.nan)}

    dbar = d_t.mean()
    lag = max(h - 1, 0)

    gamma0 = np.mean((d_t - dbar) * (d_t - dbar))
    var = gamma0

    for L in range(1, lag + 1):
        gammaL = np.mean((d_t[L:] - dbar) * (d_t[:-L] - dbar))
        weight = 1 - L/(lag + 1)
        var += 2 * weight * gammaL

    dm = dbar / np.sqrt(var / T)
    p = 2 * (1 - norm.cdf(np.abs(dm)))
    return {"DM": float(dm), "p_value": float(p), "T": int(T), "mean_gain": float(dbar)}


def dm_global(dfH, H):
    df2 = dfH.copy()
    df2["L_model"] = np.abs(df2["y_true"] - df2["y_pred"])
    df2["L_base"]  = np.abs(df2["y_true"] - df2["y_base"])

    # agrega por timestamp (evita “inflar” o T pela dimensão espacial)
    agg = df2.groupby("datetime")[["L_model", "L_base"]].mean().reset_index()
    d_t = (agg["L_base"] - agg["L_model"]).values
    return dm_test_hac(d_t, h=H)


# ==========================================================
# Treino + predição (1 rodada)
# ==========================================================
def train_predict_tcn(Xtr, Ytr, Xva, Yva, Xte,
                      filters=64, kernel_size=3, dilations=(1,2,4,8,16),
                      dropout=0.15, lr=1e-3, epochs=EPOCHS):
    tf.keras.backend.clear_session()

    model = build_tcn_model(
        lookback=Xtr.shape[1],
        n_features=Xtr.shape[2],
        h_max=Ytr.shape[1],
        filters=filters,
        kernel_size=kernel_size,
        dilations=dilations,
        dropout=dropout,
        lr=lr
    )

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    model.fit(
        Xtr, Ytr,
        validation_data=(Xva, Yva),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=1
    )

    Pte = model.predict(Xte, batch_size=BATCH_SIZE, verbose=0).astype(np.float32)
    return model, Pte


# ==========================================================
# Runner principal: data -> sequências -> TCN -> métricas -> ablation -> stats
# ==========================================================
def run(csv_path):
    # 1) Ler CSV
    dfh = pd.read_csv(csv_path)

    # padroniza nomes (se vier do export do GEE)
    if "u_component_of_wind_10m" in dfh.columns:
        dfh = dfh.rename(columns={
            "u_component_of_wind_10m": "u10",
            "v_component_of_wind_10m": "v10"
        })

    dfh["datetime"] = pd.to_datetime(dfh["datetime"])
    dfh["id"] = pd.to_numeric(dfh["id"], errors="coerce").astype(int)

    # 2) Wind speed 80m (se não tiver)
    if "wind_speed_80m" not in dfh.columns:
        dfh["wind_speed_10m"] = np.sqrt(dfh["u10"]**2 + dfh["v10"]**2)
        dfh["wind_speed_80m"] = ajustar_velocidade_altura(dfh["wind_speed_10m"].values, 10, 80, 0.1)

    # 3) Features temporais
    dfh = dfh.sort_values(["id", "datetime"]).reset_index(drop=True)
    dfh = add_time_features(dfh)

    FEATURES = ["u10", "v10", "elevation", "slope", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    TARGET = "wind_speed_80m"

    # 4) Split
    mtr, mva, mte = split_masks_hourly(dfh)
    train_df = dfh.loc[mtr].copy()
    val_df   = dfh.loc[mva].copy()
    test_df  = dfh.loc[mte].copy()

    # 5) Normalização (fit no treino)
    mu, sd, mu_y, sd_y = fit_scalers(train_df, FEATURES, TARGET)

    # 6) Sequências multi-horizonte (não seq2seq)
    tr = make_multiH_sequences_per_id(train_df, LOOKBACK, H_MAX, FEATURES, TARGET, mu, sd, mu_y, sd_y)
    va = make_multiH_sequences_per_id(val_df,   LOOKBACK, H_MAX, FEATURES, TARGET, mu, sd, mu_y, sd_y)
    te = make_multiH_sequences_per_id(test_df,  LOOKBACK, H_MAX, FEATURES, TARGET, mu, sd, mu_y, sd_y)

    if tr is None or va is None or te is None:
        raise ValueError("Sem sequências suficientes. Verifique continuidade por turbina e o período do CSV.")

    Xtr, Ytr, t0_tr, id_tr, ytr_t = tr
    Xva, Yva, t0_va, id_va, yva_t = va
    Xte, Yte, t0_te, id_te, yte_t = te

    print("\nShapes:")
    print("  Train:", Xtr.shape, Ytr.shape)
    print("  Val:  ", Xva.shape, Yva.shape)
    print("  Test: ", Xte.shape, Yte.shape)

    # ======================================================
    # 7) Modelo principal (TCN base)
    # ======================================================
    print("\n====================")
    print("TCN BASE (multi-horizonte)")
    print("====================")

    base_cfg = {"name": "TCN_base_dilated", "filters": 64, "kernel": 3, "dilations": (1,2,4,8,16), "dropout": 0.15}
    model_base, Pte_base = train_predict_tcn(
        Xtr, Ytr, Xva, Yva, Xte,
        filters=base_cfg["filters"],
        kernel_size=base_cfg["kernel"],
        dilations=base_cfg["dilations"],
        dropout=base_cfg["dropout"],
        lr=1e-3,
        epochs=EPOCHS
    )

    # métricas (TEST) por horizonte
    Yte_real = denorm(Yte, mu_y, sd_y)
    Pte_real = denorm(Pte_base, mu_y, sd_y)

    df_eval = evaluate_horizons(Yte_real, Pte_real, yte_t, H_eval=H_EVAL)
    print("\n--- Métricas (TEST): TCN vs Persistência ---")
    print(df_eval.to_string(index=False))

    # ======================================================
    # 8) Ablation Study (TEST)
    # ======================================================
    print("\n====================")
    print("ABLATION STUDY (TCN)")
    print("====================")

    ABLATIONS = [
        {"name":"TCN_base_dilated", "filters":64, "kernel":3, "dilations":(1,2,4,8,16),  "dropout":0.15},
        {"name":"TCN_no_dilation",  "filters":64, "kernel":3, "dilations":(1,1,1,1,1),   "dropout":0.15},
        {"name":"TCN_shallow",      "filters":64, "kernel":3, "dilations":(1,2,4),        "dropout":0.15},
        {"name":"TCN_deep",         "filters":64, "kernel":3, "dilations":(1,2,4,8,16,32),"dropout":0.15},
        {"name":"TCN_small",        "filters":32, "kernel":3, "dilations":(1,2,4,8,16),  "dropout":0.15},
        {"name":"TCN_large",        "filters":96, "kernel":3, "dilations":(1,2,4,8,16),  "dropout":0.15},
    ]

    ablation_rows = []
    best_name = None
    best_score = np.inf
    best_pred = None
    best_cfg = None

    for cfg in ABLATIONS:
        print(f"\n--- Rodando: {cfg['name']} ---")
        _, Pte_n = train_predict_tcn(
            Xtr, Ytr, Xva, Yva, Xte,
            filters=cfg["filters"],
            kernel_size=cfg["kernel"],
            dilations=cfg["dilations"],
            dropout=cfg["dropout"],
            lr=1e-3,
            epochs=EPOCHS
        )

        Pte_r = denorm(Pte_n, mu_y, sd_y)

        dfm = evaluate_horizons(Yte_real, Pte_r, yte_t, H_eval=H_EVAL)
        dfm["model_name"] = cfg["name"]
        dfm["filters"] = cfg["filters"]
        dfm["dilations"] = str(cfg["dilations"])
        ablation_rows.append(dfm)

        # critério simples de “melhor”: menor MAE em t+6 (pode trocar)
        score = float(dfm.loc[dfm["H_horas"] == 6, "MODEL_MAE_TEST"].values[0])
        if score < best_score:
            best_score = score
            best_name = cfg["name"]
            best_pred = Pte_n.copy()
            best_cfg = cfg

    df_ablation = pd.concat(ablation_rows, ignore_index=True)
    print("\n--- ABLATION (ordenado por H e MAE) ---")
    print(df_ablation.sort_values(["H_horas", "MODEL_MAE_TEST"])[
        ["model_name","H_horas","MODEL_MAE_TEST","BASE_MAE_TEST","GAIN_MAE","filters","dilations"]
    ].to_string(index=False))

    print(f"\n>>> Melhor (critério: menor MAE em t+6): {best_name} | cfg={best_cfg}")

    # ======================================================
    # 9) Wilcoxon + DM (melhor TCN vs persistência)
    # ======================================================
    print("\n====================")
    print("TESTES ESTATÍSTICOS (melhor TCN vs Persistência)")
    print("====================")

    stats_rows = []
    for H in H_EVAL:
        dfH = make_test_df_for_stats(id_te, t0_te, Yte, best_pred, yte_t, mu_y, sd_y, H)

        w_point = wilcoxon_pointwise(dfH)
        _, w_turb = wilcoxon_by_turbine(dfH)
        dm = dm_global(dfH, H)

        stats_rows.append({
            "H_horas": H,
            "Wilcoxon_point_p": w_point["p_value"],
            "Wilcoxon_point_median_gain_absErr": w_point["median_gain"],
            "Wilcoxon_turb_p":  w_turb["p_value"],
            "Wilcoxon_turb_median_gain_MAE": w_turb["median_gain"],
            "DM_stat": dm["DM"],
            "DM_p": dm["p_value"],
            "DM_T": dm["T"],
            "DM_mean_gain_absErr": dm["mean_gain"],
        })

    df_stats = pd.DataFrame(stats_rows)
    print("\n--- Stats (melhor TCN vs persistência) ---")
    print(df_stats.to_string(index=False))

    # robustez: % turbinas que melhoraram (GAIN_MAE>0) por horizonte
    print("\n====================")
    print("ROBUSTEZ ESPACIAL (quantas turbinas melhoraram?)")
    print("====================")
    rob_rows = []
    for H in H_EVAL:
        dfH = make_test_df_for_stats(id_te, t0_te, Yte, best_pred, yte_t, mu_y, sd_y, H)
        g = dfH.groupby("id").apply(lambda x: pd.Series({
            "mae_model": np.mean(np.abs(x["y_true"] - x["y_pred"])),
            "mae_base":  np.mean(np.abs(x["y_true"] - x["y_base"]))
        })).reset_index()
        g["gain"] = g["mae_base"] - g["mae_model"]
        rob_rows.append({
            "H_horas": H,
            "n_turbinas": int(g["id"].nunique()),
            "pct_melhorou": float(100.0 * (g["gain"] > 0).mean()),
            "gain_medio_MAE": float(g["gain"].mean())
        })
    df_rob = pd.DataFrame(rob_rows)
    print(df_rob.to_string(index=False))

    print("\nOK. (Fim do train_hourly.py)")


# ==========================================================
# Main
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Caminho do CSV horário exportado do GEE")
    args = parser.parse_args()

    run(args.csv)

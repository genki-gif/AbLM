"""
04_benchmark_classification.py
FLAb2 ベンチマークデータセットを使い、抗体言語モデルの埋め込みを下流分類タスクで評価する。

5 モデル × 2 プーリング (full-seq / CDR-H3) × 2 分類器 (SVM / XGBoost) で
accuracy, AUC-ROC, F1 (macro) をStratified 5-fold CVで測定する。

入力:
    benchmarks/flab2/repo/data/ 以下の CSV ファイル

出力:
    data/benchmarks/<dataset_name>/<model>_{seq,cdr3}_vecs.npy  (埋め込みキャッシュ)
    data/benchmarks/classification_results.csv                   (全結果)

使用方法:
    KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/04_benchmark_classification.py
    KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/04_benchmark_classification.py --datasets jain2017_tm --model antiberty
    KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/04_benchmark_classification.py --datasets all --model all --binarize quartile
"""

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_BASE = PROJECT_ROOT / "benchmarks" / "flab2" / "repo" / "data"

# ──────────────────────────────────────────────
# データセットレジストリ
# ──────────────────────────────────────────────

BENCHMARK_DATASETS = {
    "koenig2017_kd": {
        "path": "binding/koenig2017mutational_kd_g6.csv",
        "category": "binding",
        "fitness_col": "fitness",
        "binary": False,
    },
    "kirby2024_binary": {
        "path": "binding/kirby2024retrospective_ab-SARSCoV2_binary_kd.csv",
        "category": "binding",
        "fitness_col": "KD [bind/no bind]",
        "binary": True,
    },
    "peterson2024_binary": {
        "path": "binding/peterson2024integrated_ab_H1HA_binary.csv",
        "category": "binding",
        "fitness_col": "fitness",
        "binary": True,
    },
    "warszawski2019_kd": {
        "path": "binding/warszawski2019_d44_Kd.csv",
        "category": "binding",
        "fitness_col": "fitness",
        "binary": False,
    },
    "koenig2017_er": {
        "path": "expression/koenig2017mutational_er_g6.csv",
        "category": "expression",
        "fitness_col": "fitness",
        "binary": False,
    },
    "jain2017_hek": {
        "path": "expression/jain2017biophysical_HEK.csv",
        "category": "expression",
        "fitness_col": "fitness",
        "binary": False,
    },
    "marks2021_immuno": {
        "path": "immunogenicity/marks2021humanization_immunogenicity.csv",
        "category": "immunogenicity",
        "fitness_col": "fitness",
        "binary": False,
    },
    "jain2017_tm": {
        "path": "thermostability/jain2017biophysical_Tm.csv",
        "category": "thermostability",
        "fitness_col": "fitness",
        "binary": False,
    },
    "jain2017_hicrt": {
        "path": "aggregation/jain2017biophyscial_HICRT.csv",
        "category": "aggregation",
        "fitness_col": "fitness",
        "binary": False,
    },
    "jain2017_psr": {
        "path": "polyreactivity/jain2017biophysical_PSR.csv",
        "category": "polyreactivity",
        "fitness_col": "fitness",
        "binary": False,
    },
    "jain2023_fcrn": {
        "path": "pharmacokinetics/jain2023identifying_FcRnRelRT3.csv",
        "category": "pharmacokinetics",
        "fitness_col": "fitness",
        "binary": False,
    },
}

# ──────────────────────────────────────────────
# CDR3 インデックス抽出 (01_download_data.py からコピー)
# ──────────────────────────────────────────────

def get_cdr3_indices_abnumber(vh_seq: str) -> tuple[int, int] | None:
    """AbNumber を使って CDR3 位置を特定する。"""
    from abnumber import Chain
    try:
        chain = Chain(vh_seq, scheme="imgt", assign_germline=False)
        cdr3 = chain.cdr3_seq
        if cdr3 and len(cdr3) >= 3:
            idx = vh_seq.find(cdr3)
            if idx >= 0:
                return idx, idx + len(cdr3)
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────
# 02_generate_embeddings.py からのインポート
# ──────────────────────────────────────────────

def _import_embedding_module():
    """scripts/02_generate_embeddings.py をモジュールとして読み込む。"""
    script_path = PROJECT_ROOT / "scripts" / "02_generate_embeddings.py"
    spec = importlib.util.spec_from_file_location("gen_emb", str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ──────────────────────────────────────────────
# データ読み込み
# ──────────────────────────────────────────────

def load_benchmark(
    name: str,
    binarize: str = "median",
) -> tuple[list[str], list[tuple[int, int] | None], np.ndarray, int]:
    """ベンチマークデータセットを読み込み、配列・CDR3インデックス・ラベルを返す。

    Returns:
        vh_seqs, cdr3_indices, labels (binary ndarray), n_original
    """
    info = BENCHMARK_DATASETS[name]
    csv_path = BENCHMARK_BASE / info["path"]
    if not csv_path.exists():
        raise FileNotFoundError(f"ベンチマーク CSV が見つかりません: {csv_path}")

    df = pd.read_csv(csv_path)
    fitness_col = info["fitness_col"]

    # heavy と fitness が有効な行のみ
    df = df.dropna(subset=["heavy", fitness_col])
    df = df[df["heavy"].str.len() > 0].reset_index(drop=True)
    n_original = len(df)

    vh_seqs = df["heavy"].tolist()

    # CDR3 インデックス抽出
    log.info(f"  CDR3 インデックス抽出中 ({n_original} 配列) ...")
    from tqdm import tqdm
    cdr3_indices = []
    for seq in tqdm(vh_seqs, desc=f"  CDR3 ({name})", leave=False):
        cdr3_indices.append(get_cdr3_indices_abnumber(seq))

    # ラベル生成
    fitness = df[fitness_col].values.astype(float)
    if info["binary"]:
        labels = fitness.astype(int)
    else:
        if binarize == "median":
            threshold = np.median(fitness)
            labels = (fitness >= threshold).astype(int)
        elif binarize == "quartile":
            q25, q75 = np.percentile(fitness, [25, 75])
            mask = (fitness <= q25) | (fitness >= q75)
            df = df[mask].reset_index(drop=True)
            vh_seqs = [vh_seqs[i] for i in range(len(mask)) if mask[i]]
            cdr3_indices = [cdr3_indices[i] for i in range(len(mask)) if mask[i]]
            fitness = fitness[mask]
            labels = (fitness >= q75).astype(int)
        else:
            raise ValueError(f"不明な binarize 方法: {binarize}")

    log.info(f"  ラベル分布: 0={int((labels == 0).sum())}, 1={int((labels == 1).sum())}")
    return vh_seqs, cdr3_indices, labels, n_original


# ──────────────────────────────────────────────
# 埋め込み生成
# ──────────────────────────────────────────────

MODEL_CONFIGS = {
    "antiberty": {"func": "run_antiberty", "needs_device": False},
    "ablang": {"func": "run_ablang", "needs_device": False},
    "ablang2heavy": {"func": "run_ablang2_heavy", "needs_device": True},
    "esm2": {"func": "run_esm2", "needs_device": True},
    "progen2": {"func": "run_progen2", "needs_device": True},
}

MODEL_FILE_PREFIXES = {
    "antiberty": "antiberty",
    "ablang": "ablang",
    "ablang2heavy": "ablang2_heavy",
    "esm2": "esm2",
    "progen2": "progen2",
}


def generate_embeddings(
    model_name: str,
    vh_seqs: list[str],
    cdr3_indices: list[tuple[int, int] | None],
    batch_size: int,
    device: str,
    out_dir: Path,
    emb_mod,
) -> None:
    """指定モデルの埋め込みを生成する。"""
    config = MODEL_CONFIGS[model_name]
    func = getattr(emb_mod, config["func"])

    if model_name == "progen2":
        # ProGen-2 は batch_size 引数を取らない
        func(vh_seqs, cdr3_indices, device, out_dir)
    elif config["needs_device"]:
        func(vh_seqs, cdr3_indices, batch_size, device, out_dir)
    else:
        func(vh_seqs, cdr3_indices, batch_size, out_dir)


# ──────────────────────────────────────────────
# 分類パイプライン
# ──────────────────────────────────────────────

def run_classification(
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
) -> list[dict]:
    """SVM と XGBoost で分類し、結果をリストで返す。"""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = ["accuracy", "roc_auc", "f1_macro"]
    results = []

    # SVM
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, random_state=42)),
    ])
    svm_scores = cross_validate(svm_pipe, X, y, cv=cv, scoring=scoring)
    results.append({
        "classifier": "SVM",
        "accuracy": svm_scores["test_accuracy"].mean(),
        "accuracy_std": svm_scores["test_accuracy"].std(),
        "auc_roc": svm_scores["test_roc_auc"].mean(),
        "auc_roc_std": svm_scores["test_roc_auc"].std(),
        "f1_macro": svm_scores["test_f1_macro"].mean(),
        "f1_macro_std": svm_scores["test_f1_macro"].std(),
    })

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_scores = cross_validate(xgb, X, y, cv=cv, scoring=scoring)
    results.append({
        "classifier": "XGBoost",
        "accuracy": xgb_scores["test_accuracy"].mean(),
        "accuracy_std": xgb_scores["test_accuracy"].std(),
        "auc_roc": xgb_scores["test_roc_auc"].mean(),
        "auc_roc_std": xgb_scores["test_roc_auc"].std(),
        "f1_macro": xgb_scores["test_f1_macro"].mean(),
        "f1_macro_std": xgb_scores["test_f1_macro"].std(),
    })

    return results


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FLAb2 ベンチマークで抗体言語モデルの埋め込みを分類評価する"
    )
    parser.add_argument("--batch", type=int, default=32, help="バッチサイズ (default: 32)")
    parser.add_argument(
        "--device", type=str, default="auto",
        help="使用デバイス: auto / cpu / cuda (default: auto)",
    )
    parser.add_argument(
        "--model", type=str, default="all",
        help="評価するモデル: all / antiberty / ablang / ablang2heavy / esm2 / progen2 (default: all)",
    )
    parser.add_argument(
        "--datasets", type=str, default="all",
        help="評価するデータセット: all / カンマ区切りID (default: all)",
    )
    parser.add_argument(
        "--binarize", type=str, default="median", choices=["median", "quartile"],
        help="連続値の二値化方法: median / quartile (default: median)",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="交差検証のフォールド数 (default: 5)",
    )
    args = parser.parse_args()

    import torch
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    log.info(f"使用デバイス: {device}")

    # モデル選定
    if args.model == "all":
        models = list(MODEL_CONFIGS.keys())
    else:
        models = [args.model]
        for m in models:
            if m not in MODEL_CONFIGS:
                log.error(f"不明なモデル: {m}. 選択肢: {list(MODEL_CONFIGS.keys())}")
                sys.exit(1)

    # データセット選定
    if args.datasets == "all":
        dataset_names = list(BENCHMARK_DATASETS.keys())
    else:
        dataset_names = [d.strip() for d in args.datasets.split(",")]
        for d in dataset_names:
            if d not in BENCHMARK_DATASETS:
                log.error(f"不明なデータセット: {d}. 選択肢: {list(BENCHMARK_DATASETS.keys())}")
                sys.exit(1)

    # 埋め込みモジュール読み込み
    log.info("02_generate_embeddings.py を読み込み中 ...")
    emb_mod = _import_embedding_module()

    results_dir = PROJECT_ROOT / "data" / "benchmarks"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "classification_results.csv"

    # 既存の結果を読み込み（インクリメンタル保存: 途中再開に対応）
    if out_csv.exists():
        existing_df = pd.read_csv(out_csv)
        all_results = existing_df.to_dict("records")
        done_keys = set(
            zip(existing_df["dataset"], existing_df["model"],
                existing_df["pooling"], existing_df["classifier"])
        )
        log.info(f"既存の結果を読み込み: {len(all_results)} 行 ({out_csv})")
    else:
        all_results = []
        done_keys = set()

    for ds_name in dataset_names:
        ds_info = BENCHMARK_DATASETS[ds_name]
        log.info(f"\n{'='*60}")
        log.info(f"データセット: {ds_name} ({ds_info['category']})")
        log.info(f"{'='*60}")

        # データ読み込み
        vh_seqs, cdr3_indices, labels, n_original = load_benchmark(
            ds_name, binarize=args.binarize,
        )
        n_samples = len(labels)

        # 出力ディレクトリ
        ds_out_dir = results_dir / ds_name
        ds_out_dir.mkdir(parents=True, exist_ok=True)

        for model_name in models:
            prefix = MODEL_FILE_PREFIXES[model_name]
            log.info(f"\n--- {model_name} ---")

            # 埋め込み生成
            generate_embeddings(
                model_name, vh_seqs, cdr3_indices,
                args.batch, device, ds_out_dir, emb_mod,
            )

            # 分類評価（seq と cdr3 の2レベル）
            for pooling in ["seq", "cdr3"]:
                # 既に完了済みならスキップ
                if all((ds_name, model_name, pooling, clf) in done_keys
                       for clf in ["SVM", "XGBoost"]):
                    log.info(f"  スキップ（結果済み）: {model_name} / {pooling}")
                    continue

                vec_path = ds_out_dir / f"{prefix}_{pooling}_vecs.npy"
                if not vec_path.exists():
                    log.warning(f"  埋め込みファイルが見つかりません: {vec_path}")
                    continue

                X = np.load(vec_path)
                if X.shape[0] != n_samples:
                    log.warning(
                        f"  サンプル数不一致: 埋め込み={X.shape[0]}, ラベル={n_samples}. スキップ"
                    )
                    continue

                if len(np.unique(labels)) < 2:
                    log.warning(f"  クラスが1つしかないためスキップ: {ds_name} / {model_name} / {pooling}")
                    continue

                log.info(f"  分類中: {model_name} / {pooling} / X={X.shape}")
                clf_results = run_classification(X, labels, args.cv_folds)

                for r in clf_results:
                    all_results.append({
                        "dataset": ds_name,
                        "category": ds_info["category"],
                        "n_samples": n_samples,
                        "model": model_name,
                        "pooling": pooling,
                        **r,
                    })

                # インクリメンタル保存
                pd.DataFrame(all_results).to_csv(out_csv, index=False)

    # 最終サマリー表示
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(out_csv, index=False)
        log.info(f"\n結果を保存しました: {out_csv} ({len(results_df)} 行)")

        print("\n" + "=" * 100)
        print("分類結果サマリー")
        print("=" * 100)
        summary_cols = [
            "dataset", "category", "n_samples", "model", "pooling", "classifier",
            "accuracy", "auc_roc", "f1_macro",
        ]
        print(results_df[summary_cols].to_string(index=False, float_format="%.4f"))
    else:
        log.warning("結果がありません。")


if __name__ == "__main__":
    main()

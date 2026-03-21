"""
03_compute_pgen_pll.py
OLGA による VDJ 生成確率 (Pgen) と各抗体言語モデルの疑似対数尤度 (PLL) を計算して保存する。

入力:
    data/sequences_processed.csv   (01_download_data.py の出力)

出力 (data/ フォルダ):
    pgen.npy                   OLGA Pgen (条件なし)            (N,) float64
    pgen_vj.npy                OLGA Pgen (V/J 条件付き)        (N,) float64
    antiberty_pll.npy          AntiBERTy 全配列 PLL            (N,) float32
    antiberty_cdr3_pll.npy     AntiBERTy CDR3-specific PLL    (N,) float32
    ablang_pll.npy             AbLang 全配列 PLL               (N,) float32
    ablang_cdr3_pll.npy        AbLang CDR3-specific PLL       (N,) float32
    ablang2_paired_pll.npy     AbLang-2 (paired) 全配列 PLL   (N,) float32
    ablang2_paired_cdr3_pll.npy AbLang-2 (paired) CDR3 PLL   (N,) float32
    ablang2_heavy_pll.npy      AbLang-2 (heavy) 全配列 PLL    (N,) float32
    ablang2_heavy_cdr3_pll.npy  AbLang-2 (heavy) CDR3 PLL    (N,) float32
    pgen_pll_scores.csv        全スコアをまとめた CSV

PLL 定義:
    Full-seq PLL  : VH 全残基をマスク対象として平均した疑似対数尤度
    CDR3-spec PLL : VH 全体をモデルに入力しつつ CDR3 残基のみをマスク対象とし、
                    CDR3 位置のみで平均した疑似対数尤度

OLGA CDR3 フォーマット:
    OAS cdr3_aa_heavy は保存残基 C (VH の直前) と W (直後) を含まない。
    OLGA の定義は C inclusive → W inclusive なので
    cdr3_olga = 'C' + cdr3_aa_heavy + vh_seq[cdr3_end]  で変換する。

使用方法:
    KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/03_compute_pgen_pll.py
    KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/03_compute_pgen_pll.py --model pgen
    KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/03_compute_pgen_pll.py --model antiberty --batch 16
"""

import argparse
import ast
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

SCORE_COLS = [
    "pgen",
    "pgen_vj",
    "antiberty_pll",
    "antiberty_cdr3_pll",
    "ablang_pll",
    "ablang_cdr3_pll",
    "ablang2_paired_pll",
    "ablang2_paired_cdr3_pll",
    "ablang2_heavy_pll",
    "ablang2_heavy_cdr3_pll",
]


# ──────────────────────────────────────────────
# OLGA Pgen
# ──────────────────────────────────────────────

def run_pgen(df: pd.DataFrame, out_dir: Path) -> None:
    """OLGA humanIGH モデルで Pgen (条件なし + V/J 条件付き) を計算する。"""
    pgen_path = out_dir / "pgen.npy"
    pgen_vj_path = out_dir / "pgen_vj.npy"
    if pgen_path.exists() and pgen_vj_path.exists():
        log.info("OLGA Pgen: キャッシュを検出、スキップ")
        return

    import os
    import olga
    import olga.load_model as load_model
    import olga.generation_probability as pgen_module

    model_dir = os.path.join(
        os.path.dirname(olga.__file__), "default_models", "human_B_heavy"
    )
    log.info(f"OLGA: モデルディレクトリ = {model_dir}")

    genomic_data = load_model.GenomicDataVDJ()
    genomic_data.load_igor_genomic_data(
        os.path.join(model_dir, "model_params.txt"),
        os.path.join(model_dir, "V_gene_CDR3_anchors.csv"),
        os.path.join(model_dir, "J_gene_CDR3_anchors.csv"),
    )
    generative_model = load_model.GenerativeModelVDJ()
    generative_model.load_and_process_igor_model(
        os.path.join(model_dir, "model_marginals.txt")
    )
    pgen_model = pgen_module.GenerationProbabilityVDJ(generative_model, genomic_data)
    log.info("OLGA Pgen: モデル読み込み完了。計算開始 ...")

    pgens: list[float] = []
    pgens_vj: list[float] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="OLGA Pgen"):
        cdr3 = str(row["cdr3_aa_heavy"])
        v_call = str(row["v_call_heavy"])
        j_call = str(row["j_call_heavy"])
        vh = str(row["vh_seq"])

        # OLGA CDR3 = 'C' + cdr3_aa_heavy + 終端残基 (VH 配列の CDR3 直後)
        try:
            idx = ast.literal_eval(str(row["cdr3_indices"]))
            terminal = vh[idx[1]] if idx is not None and idx[1] < len(vh) else "W"
        except Exception:
            terminal = "W"
        cdr3_olga = "C" + cdr3 + terminal

        # 条件なし Pgen
        try:
            p = pgen_model.compute_aa_CDR3_pgen(cdr3_olga)
            pgens.append(float(p) if p > 0 else np.nan)
        except Exception:
            pgens.append(np.nan)

        # V/J 条件付き Pgen
        try:
            p_vj = pgen_model.compute_aa_CDR3_pgen(cdr3_olga, v_call, j_call)
            pgens_vj.append(float(p_vj) if p_vj > 0 else np.nan)
        except Exception:
            pgens_vj.append(np.nan)

    np.save(pgen_path, np.array(pgens, dtype=np.float64))
    np.save(pgen_vj_path, np.array(pgens_vj, dtype=np.float64))
    log.info(f"OLGA Pgen: 保存完了 → {pgen_path}, {pgen_vj_path}")

    valid = np.sum(~np.isnan(pgens))
    valid_vj = np.sum(~np.isnan(pgens_vj))
    log.info(f"  Pgen 有効件数: {valid}/{len(pgens)}, Pgen_VJ 有効件数: {valid_vj}/{len(pgens_vj)}")


# ──────────────────────────────────────────────
# AntiBERTy PLL
# ──────────────────────────────────────────────

def _antiberty_cdr3_pll_single(
    runner,
    vh: str,
    cdr3_start: int,
    cdr3_end: int,
    inner_batch: int = 32,
) -> float:
    """AntiBERTy で CDR3 残基のみの masked PLL を計算する。

    各 CDR3 位置 p に対して masked sequence を作成し、
    その位置の true AA の log-likelihood を平均する。
    """
    positions = list(range(cdr3_start, cdr3_end))
    if not positions:
        return float("nan")

    masked_sequences = []
    for p in positions:
        masked = list(vh[:p]) + ["[MASK]"] + list(vh[p + 1:])
        masked_sequences.append(" ".join(masked))

    tokenizer_out = runner.tokenizer(
        masked_sequences, return_tensors="pt", padding=True
    )
    tokens = tokenizer_out["input_ids"].to(runner.device)
    attention_mask = tokenizer_out["attention_mask"].to(runner.device)

    all_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(masked_sequences), inner_batch):
            b_end = min(i + inner_batch, len(masked_sequences))
            outputs = runner.model(
                input_ids=tokens[i:b_end],
                attention_mask=attention_mask[i:b_end],
            )
            all_logits.append(outputs.prediction_logits.cpu())

    # (cdr3_len, seq_len+2, vocab_size)
    all_logits_t = torch.cat(all_logits, dim=0)
    all_logits_t[:, :, runner.tokenizer.all_special_ids] = -float("inf")
    all_logits_t = all_logits_t[:, 1:-1, :]  # CLS/SEP を除去 → (cdr3_len, seq_len, vocab)

    # 各 CDR3 位置 p の logit を抽出
    cdr3_logits = torch.stack(
        [all_logits_t[i, p, :] for i, p in enumerate(positions)]
    )  # (cdr3_len, vocab)

    # True ラベル
    labels_full = runner.tokenizer.encode(
        " ".join(list(vh)), return_tensors="pt"
    )[:, 1:-1]  # (1, seq_len)
    cdr3_labels = labels_full[0, cdr3_start:cdr3_end]  # (cdr3_len,)

    nll = F.cross_entropy(cdr3_logits, cdr3_labels, reduction="mean")
    return -nll.item()


def run_antiberty_pll(
    vh_seqs: list[str],
    cdr3_indices: list,
    inner_batch: int,
    device: str,
    out_dir: Path,
) -> None:
    full_path = out_dir / "antiberty_pll.npy"
    cdr3_path = out_dir / "antiberty_cdr3_pll.npy"
    if full_path.exists() and cdr3_path.exists():
        log.info("AntiBERTy PLL: キャッシュを検出、スキップ")
        return

    from antiberty import AntiBERTyRunner

    log.info("AntiBERTy PLL: モデル読み込み中 ...")
    runner = AntiBERTyRunner()
    if device != "cpu":
        runner.model.to(device)
        runner.device = torch.device(device)
        log.info(f"AntiBERTy PLL: モデルを {device} に移動")

    full_plls: list[float] = []
    cdr3_plls: list[float] = []

    for vh, idx in tqdm(
        zip(vh_seqs, cdr3_indices), total=len(vh_seqs), desc="AntiBERTy PLL"
    ):
        # 全配列 PLL (GPU 対応のカスタム実装を使用)
        full_pll = _antiberty_cdr3_pll_single(runner, vh, 0, len(vh), inner_batch)
        full_plls.append(full_pll)

        # CDR3-specific PLL
        if idx is not None:
            start, end = idx
            cdr3_pll = _antiberty_cdr3_pll_single(runner, vh, start, end, inner_batch)
        else:
            cdr3_pll = float("nan")
        cdr3_plls.append(cdr3_pll)

    np.save(full_path, np.array(full_plls, dtype=np.float32))
    np.save(cdr3_path, np.array(cdr3_plls, dtype=np.float32))
    log.info(f"AntiBERTy PLL: 保存完了 → {full_path}, {cdr3_path}")


# ──────────────────────────────────────────────
# AbLang PLL (v1, heavy chain)
# ──────────────────────────────────────────────

def _ablang_masked_pll(
    ab_model,
    tok,
    vh: str,
    positions: list[int],
    device,
    inner_batch: int = 32,
) -> float:
    """AbLang で指定残基位置の masked PLL を計算する。

    AbLang トークン構造: [<(0), aa1(1), ..., aaL(L), >(L+1)]
    残基 i の token index = i + 1

    positions: VH 上の 0-indexed 残基位置のリスト
    """
    if not positions:
        return float("nan")

    n_pos = len(positions)
    base_tokens = tok([vh], pad=True)  # (1, L+2), CPU tensor

    # n_pos 個のマスク済みトークン行列を作成: (n_pos, L+2)
    all_masked = base_tokens.expand(n_pos, -1).clone().to(device)
    for i, p in enumerate(positions):
        all_masked[i, p + 1] = 23  # * = mask token

    # True AA の 0-indexed in 20AA space: vocab_to_token[aa] - 1
    try:
        true_idxs = [tok.vocab_to_token[vh[p]] - 1 for p in positions]
    except KeyError as e:
        log.warning(f"AbLang: 未知のアミノ酸 {e}, スキップ")
        return float("nan")

    log_probs: list[float] = []
    with torch.no_grad():
        for i in range(0, n_pos, inner_batch):
            b_end = min(i + inner_batch, n_pos)
            logits = ab_model(all_masked[i:b_end])  # (batch, L+2, vocab_size)
            for j, p in enumerate(positions[i:b_end]):
                # tokens 1-20 が 20 種 AA に対応
                aa_logits = logits[j, p + 1, 1:21].float()
                log_p = F.log_softmax(aa_logits, dim=0)
                log_probs.append(log_p[true_idxs[i + j]].item())

    return float(np.mean(log_probs))


def run_ablang_pll(
    vh_seqs: list[str],
    cdr3_indices: list,
    inner_batch: int,
    device: str,
    out_dir: Path,
) -> None:
    full_path = out_dir / "ablang_pll.npy"
    cdr3_path = out_dir / "ablang_cdr3_pll.npy"
    if full_path.exists() and cdr3_path.exists():
        log.info("AbLang PLL: キャッシュを検出、スキップ")
        return

    import ablang

    log.info("AbLang PLL: モデル読み込み中 ...")
    model = ablang.pretrained("heavy")
    model.freeze()

    ab_model = model.AbLang
    ab_model.eval()
    if device != "cpu":
        ab_model.to(device)
        log.info(f"AbLang PLL: モデルを {device} に移動")
    tok = model.tokenizer

    full_plls: list[float] = []
    cdr3_plls: list[float] = []

    for vh, idx in tqdm(
        zip(vh_seqs, cdr3_indices), total=len(vh_seqs), desc="AbLang PLL"
    ):
        L = len(vh)
        full_positions = list(range(L))
        full_pll = _ablang_masked_pll(ab_model, tok, vh, full_positions, device, inner_batch)
        full_plls.append(full_pll)

        if idx is not None:
            start, end = idx
            cdr3_positions = list(range(start, end))
            cdr3_pll = _ablang_masked_pll(
                ab_model, tok, vh, cdr3_positions, device, inner_batch
            )
        else:
            cdr3_pll = float("nan")
        cdr3_plls.append(cdr3_pll)

    np.save(full_path, np.array(full_plls, dtype=np.float32))
    np.save(cdr3_path, np.array(cdr3_plls, dtype=np.float32))
    log.info(f"AbLang PLL: 保存完了 → {full_path}, {cdr3_path}")


# ──────────────────────────────────────────────
# AbLang-2 PLL (paired model)
# ──────────────────────────────────────────────

def _ablang2_load_model(device: str):
    import ablang2

    return ablang2.pretrained(
        model_to_use="ablang2-paired",
        random_init=False,
        ncpu=1,
        device=device,
    )


def _ablang2_masked_pll(
    ab_model,
    tok,
    vh: str,
    vl: str,
    positions: list[int],
    device: str = "cpu",
    inner_batch: int = 16,
) -> float:
    """AbLang-2 で指定 VH 残基位置の masked PLL を計算する。

    w_extra_tkns=True のトークン構造:
        [<(0), VH[0], ..., VH[L-1], >(L), |(L+1), <(L+2), VL..., >]
    VH 残基 i の token index = i + 1

    positions: VH 上の 0-indexed 残基位置のリスト
    vl: 空文字列 "" の場合は VH 単独入力 (heavy-only モード)
    """
    if not positions:
        return float("nan")

    n_pos = len(positions)
    pairs = [[vh, vl]]
    base_tokens = tok(pairs, pad=True, w_extra_tkns=True, device=device)  # (1, L_full)

    # n_pos 個のマスク済みトークン行列: (n_pos, L_full)
    all_masked = base_tokens.expand(n_pos, -1).clone()
    for i, p in enumerate(positions):
        all_masked[i, p + 1] = tok.mask_token  # +1: start token <

    # True AA のトークン ID
    try:
        true_tokens = [tok.aa_to_token[vh[p]] for p in positions]
    except KeyError as e:
        log.warning(f"AbLang-2: 未知のアミノ酸 {e}, スキップ")
        return float("nan")

    special_set = set(tok.all_special_tokens)

    log_probs: list[float] = []
    with torch.no_grad():
        for i in range(0, n_pos, inner_batch):
            b_end = min(i + inner_batch, n_pos)
            logits = ab_model(all_masked[i:b_end])  # (batch, L_full, 26)
            for j, p in enumerate(positions[i:b_end]):
                pos_logits = logits[j, p + 1, :].float().clone()
                for sp in special_set:
                    pos_logits[sp] = -float("inf")
                log_p = F.log_softmax(pos_logits, dim=0)
                log_probs.append(log_p[true_tokens[i + j]].item())

    return float(np.mean(log_probs))


def run_ablang2_pll(
    vh_seqs: list[str],
    vl_seqs: list[str],
    cdr3_indices: list,
    inner_batch: int,
    device: str,
    out_dir: Path,
) -> None:
    """AbLang-2 (VH+VL ペア入力) の PLL を計算する。"""
    full_path = out_dir / "ablang2_paired_pll.npy"
    cdr3_path = out_dir / "ablang2_paired_cdr3_pll.npy"
    if full_path.exists() and cdr3_path.exists():
        log.info("AbLang-2 (paired) PLL: キャッシュを検出、スキップ")
        return

    log.info("AbLang-2 (paired) PLL: モデル読み込み中 ...")
    model = _ablang2_load_model(device)
    ab_model = model.AbLang
    ab_model.eval()
    tok = model.tokenizer

    full_plls: list[float] = []
    cdr3_plls: list[float] = []

    for vh, vl, idx in tqdm(
        zip(vh_seqs, vl_seqs, cdr3_indices),
        total=len(vh_seqs),
        desc="AbLang-2 paired PLL",
    ):
        L = len(vh)
        full_positions = list(range(L))
        full_pll = _ablang2_masked_pll(
            ab_model, tok, vh, vl, full_positions, device, inner_batch
        )
        full_plls.append(full_pll)

        if idx is not None:
            start, end = idx
            cdr3_positions = list(range(start, end))
            cdr3_pll = _ablang2_masked_pll(
                ab_model, tok, vh, vl, cdr3_positions, device, inner_batch
            )
        else:
            cdr3_pll = float("nan")
        cdr3_plls.append(cdr3_pll)

    np.save(full_path, np.array(full_plls, dtype=np.float32))
    np.save(cdr3_path, np.array(cdr3_plls, dtype=np.float32))
    log.info(f"AbLang-2 (paired) PLL: 保存完了 → {full_path}, {cdr3_path}")


def run_ablang2_heavy_pll(
    vh_seqs: list[str],
    cdr3_indices: list,
    inner_batch: int,
    device: str,
    out_dir: Path,
) -> None:
    """AbLang-2 (VH 単独入力, VL="") の PLL を計算する。"""
    full_path = out_dir / "ablang2_heavy_pll.npy"
    cdr3_path = out_dir / "ablang2_heavy_cdr3_pll.npy"
    if full_path.exists() and cdr3_path.exists():
        log.info("AbLang-2 (heavy-only) PLL: キャッシュを検出、スキップ")
        return

    log.info("AbLang-2 (heavy-only) PLL: モデル読み込み中 ...")
    model = _ablang2_load_model(device)
    ab_model = model.AbLang
    ab_model.eval()
    tok = model.tokenizer

    full_plls: list[float] = []
    cdr3_plls: list[float] = []

    for vh, idx in tqdm(
        zip(vh_seqs, cdr3_indices), total=len(vh_seqs), desc="AbLang-2 heavy PLL"
    ):
        L = len(vh)
        full_positions = list(range(L))
        full_pll = _ablang2_masked_pll(
            ab_model, tok, vh, "", full_positions, device, inner_batch
        )
        full_plls.append(full_pll)

        if idx is not None:
            start, end = idx
            cdr3_positions = list(range(start, end))
            cdr3_pll = _ablang2_masked_pll(
                ab_model, tok, vh, "", cdr3_positions, device, inner_batch
            )
        else:
            cdr3_pll = float("nan")
        cdr3_plls.append(cdr3_pll)

    np.save(full_path, np.array(full_plls, dtype=np.float32))
    np.save(cdr3_path, np.array(cdr3_plls, dtype=np.float32))
    log.info(f"AbLang-2 (heavy-only) PLL: 保存完了 → {full_path}, {cdr3_path}")


# ──────────────────────────────────────────────
# CSV 構築
# ──────────────────────────────────────────────

def build_csv(df: pd.DataFrame, out_dir: Path) -> None:
    """利用可能な npy ファイルを読み込んで pgen_pll_scores.csv を生成する。"""
    scores: dict[str, np.ndarray] = {}
    for col in SCORE_COLS:
        npy_path = out_dir / f"{col}.npy"
        if npy_path.exists():
            scores[col] = np.load(npy_path)

    if not scores:
        log.warning("スコアファイルが見つかりません。CSV を生成できません。")
        return

    df_scores = pd.DataFrame(scores)
    meta_cols = ["v_family", "v_call_heavy", "j_call_heavy", "cdr3_aa_heavy"]
    df_out = pd.concat(
        [df[meta_cols].reset_index(drop=True), df_scores.reset_index(drop=True)],
        axis=1,
    )
    csv_path = out_dir / "pgen_pll_scores.csv"
    df_out.to_csv(csv_path, index=False)
    log.info(f"CSV 保存完了 → {csv_path}  ({len(df_out)} 行 × {len(df_out.columns)} 列)")


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OLGA Pgen と各モデルの PLL を計算して保存する"
    )
    parser.add_argument(
        "--datadir", type=str, default="data", help="データディレクトリ (default: data)"
    )
    parser.add_argument(
        "--batch", type=int, default=32, help="内部バッチサイズ (default: 32)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="使用デバイス: auto / cpu / cuda (default: auto)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help=(
            "実行するモデル: "
            "all / pgen / antiberty / ablang / ablang2 / ablang2heavy "
            "(default: all)"
        ),
    )
    args = parser.parse_args()

    data_dir = Path(args.datadir)
    csv_path = data_dir / "sequences_processed.csv"
    if not csv_path.exists():
        log.error(
            f"{csv_path} が見つかりません。先に 01_download_data.py を実行してください。"
        )
        sys.exit(1)

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    log.info(f"使用デバイス: {device}")

    df = pd.read_csv(csv_path)
    log.info(f"読み込んだ配列数: {len(df)}")

    vh_seqs = df["vh_seq"].tolist()
    vl_seqs = df["vl_seq"].tolist()

    def parse_idx(val):
        if pd.isna(val) or val is None:
            return None
        try:
            result = ast.literal_eval(str(val))
            return tuple(result) if result is not None else None
        except Exception:
            return None

    cdr3_indices = [parse_idx(v) for v in df["cdr3_indices"]]

    models_to_run = set(
        ["pgen", "antiberty", "ablang", "ablang2", "ablang2heavy"]
        if args.model == "all"
        else [args.model]
    )

    inner_batch2 = min(args.batch, 16)  # AbLang-2 は長い配列なので小さめ

    if "pgen" in models_to_run:
        run_pgen(df, data_dir)

    if "antiberty" in models_to_run:
        run_antiberty_pll(vh_seqs, cdr3_indices, args.batch, device, data_dir)

    if "ablang" in models_to_run:
        run_ablang_pll(vh_seqs, cdr3_indices, args.batch, device, data_dir)

    if "ablang2" in models_to_run:
        run_ablang2_pll(vh_seqs, vl_seqs, cdr3_indices, inner_batch2, device, data_dir)

    if "ablang2heavy" in models_to_run:
        run_ablang2_heavy_pll(vh_seqs, cdr3_indices, inner_batch2, device, data_dir)

    build_csv(df, data_dir)
    log.info("全モデルの PLL + Pgen 計算完了")


if __name__ == "__main__":
    main()

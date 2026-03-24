"""
02_generate_embeddings.py
前処理済み配列から AntiBERTy・AbLang・AbLang-2・ESM-2・ProGen-2 の埋め込みを生成して保存する。

入力:
    data/sequences_processed.csv   (01_download_data.py の出力)

出力 (data/ フォルダ):
    antiberty_seq_vecs.npy         AntiBERTy シーケンスレベルベクトル (N, 512)
    antiberty_cdr3_vecs.npy        AntiBERTy CDR-H3 レベルベクトル   (N, 512)
    ablang_seq_vecs.npy            AbLang シーケンスレベルベクトル    (N, 768)
    ablang_cdr3_vecs.npy           AbLang CDR-H3 レベルベクトル      (N, 768)
    ablang2_seq_vecs.npy           AbLang-2 (VH+VL ペア入力) シーケンスレベル (N, D)
    ablang2_cdr3_vecs.npy          AbLang-2 (VH+VL ペア入力) CDR-H3 レベル   (N, D)
    ablang2_heavy_seq_vecs.npy     AbLang-2 (VH 単独入力)   シーケンスレベル (N, D)
    ablang2_heavy_cdr3_vecs.npy    AbLang-2 (VH 単独入力)   CDR-H3 レベル   (N, D)
    esm2_seq_vecs.npy              ESM-2 シーケンスレベルベクトル     (N, 1280)
    esm2_cdr3_vecs.npy             ESM-2 CDR-H3 レベルベクトル       (N, 1280)
    progen2_seq_vecs.npy           ProGen-2 シーケンスレベルベクトル  (N, 1024)
    progen2_cdr3_vecs.npy          ProGen-2 CDR-H3 レベルベクトル    (N, 1024)

トークン形式 (ablang2-paired):
    ペア  : [<, VH残基..., >, |, <, VL残基..., >]  → VH は index 1 〜 1+len(VH)
    VH単独: [<, VH残基..., >, |]                    → VH は index 1 〜 1+len(VH)

使用方法:
    mamba run -n ablm python scripts/02_generate_embeddings.py [--batch 32] [--device cpu]
    mamba run -n ablm python scripts/02_generate_embeddings.py --model ablang2heavy
    mamba run -n ablm python scripts/02_generate_embeddings.py --model esm2
    mamba run -n ablm python scripts/02_generate_embeddings.py --model progen2
"""

import argparse
import ast
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# プーリング共通ユーティリティ
# ──────────────────────────────────────────────

def pool_seq(residue_embs: list[np.ndarray]) -> np.ndarray:
    """全残基の平均プーリング → (N, D)。"""
    return np.stack([e.mean(axis=0) for e in residue_embs])


def pool_cdr3(
    residue_embs: list[np.ndarray],
    cdr3_indices: list[tuple[int, int] | None],
) -> np.ndarray:
    """CDR3 残基の平均プーリング → (N, D)。インデックスが無効な場合は全残基平均。"""
    out = []
    for emb, idx in zip(residue_embs, cdr3_indices):
        if idx is not None:
            start, end = idx
            cdr3_emb = emb[start:end]
            if len(cdr3_emb) > 0:
                out.append(cdr3_emb.mean(axis=0))
                continue
        out.append(emb.mean(axis=0))
    return np.stack(out)


# ──────────────────────────────────────────────
# AntiBERTy
# ──────────────────────────────────────────────

def run_antiberty(
    vh_seqs: list[str],
    cdr3_indices: list[tuple[int, int] | None],
    batch_size: int,
    out_dir: Path,
) -> None:
    from antiberty import AntiBERTyRunner

    seq_path = out_dir / "antiberty_seq_vecs.npy"
    cdr3_path = out_dir / "antiberty_cdr3_vecs.npy"
    if seq_path.exists() and cdr3_path.exists():
        log.info("AntiBERTy: キャッシュを検出、スキップ")
        return

    log.info("AntiBERTy: モデル読み込み中 ...")
    runner = AntiBERTyRunner()

    res_embs: list[np.ndarray] = []
    for i in tqdm(range(0, len(vh_seqs), batch_size), desc="AntiBERTy"):
        batch = vh_seqs[i:i + batch_size]
        embs = runner.embed(batch)  # list of Tensor (L+2, 512)
        for emb in embs:
            # index 0 = [CLS], -1 = [SEP] を除外
            res_embs.append(emb[1:-1].detach().cpu().float().numpy())

    np.save(seq_path, pool_seq(res_embs))
    np.save(cdr3_path, pool_cdr3(res_embs, cdr3_indices))
    log.info(f"AntiBERTy: 保存完了 → {seq_path}, {cdr3_path}")


# ──────────────────────────────────────────────
# AbLang
# ──────────────────────────────────────────────

def run_ablang(
    vh_seqs: list[str],
    cdr3_indices: list[tuple[int, int] | None],
    batch_size: int,
    out_dir: Path,
) -> None:
    import ablang

    seq_path = out_dir / "ablang_seq_vecs.npy"
    cdr3_path = out_dir / "ablang_cdr3_vecs.npy"
    if seq_path.exists() and cdr3_path.exists():
        log.info("AbLang: キャッシュを検出、スキップ")
        return

    log.info("AbLang: モデル読み込み中 ...")
    model = ablang.pretrained("heavy")
    model.freeze()

    seq_vecs_list: list[np.ndarray] = []
    res_embs: list[np.ndarray] = []

    for i in tqdm(range(0, len(vh_seqs), batch_size), desc="AbLang"):
        batch = vh_seqs[i:i + batch_size]

        seq_out = model(batch, mode="seqcoding")  # (B, 768)
        seq_vecs_list.append(np.asarray(seq_out, dtype=np.float32))

        res_out = model(batch, mode="rescoding")  # list of (L, 768)
        for r in res_out:
            res_embs.append(np.asarray(r, dtype=np.float32))

    np.save(seq_path, np.vstack(seq_vecs_list))
    np.save(cdr3_path, pool_cdr3(res_embs, cdr3_indices))
    log.info(f"AbLang: 保存完了 → {seq_path}, {cdr3_path}")


# ──────────────────────────────────────────────
# AbLang-2
# ──────────────────────────────────────────────

def _ablang2_load_model(device: str):
    """AbLang-2 (ablang2-paired) モデルを読み込んで返す。"""
    import ablang2
    return ablang2.pretrained(
        model_to_use="ablang2-paired",
        random_init=False,
        ncpu=1,
        device=device,
    )


def _ablang2_extract_vh(
    model,
    seqs_as_pairs: list[list[str]],
    vh_seqs_batch: list[str],
    device: str,
) -> list[np.ndarray]:
    """
    seqs_as_pairs: [[vh, vl], ...] または [[vh, ""], ...]
    トークン形式: [<, VH残基..., >, |, <, VL残基..., >]
                   index 0   1..l_vh  l_vh+1  ...
    VH 残基は必ず index 1 〜 1+len(vh) に位置する。
    """
    tokenized = model.tokenizer(seqs_as_pairs, pad=True, w_extra_tkns=True, device=device)
    with torch.no_grad():
        rescoding = model.AbRep(tokenized).last_hidden_states  # (B, L_padded, D)
    res_embs = []
    for j, vh in enumerate(vh_seqs_batch):
        l_vh = len(vh)
        vh_emb = rescoding[j, 1:1 + l_vh, :].detach().cpu().float().numpy()
        res_embs.append(vh_emb)
    return res_embs


def run_ablang2(
    vh_seqs: list[str],
    vl_seqs: list[str],
    cdr3_indices: list[tuple[int, int] | None],
    batch_size: int,
    device: str,
    out_dir: Path,
) -> None:
    """AbLang-2 (VH+VL ペア入力) の埋め込みを生成する。"""
    seq_path = out_dir / "ablang2_seq_vecs.npy"
    cdr3_path = out_dir / "ablang2_cdr3_vecs.npy"
    if seq_path.exists() and cdr3_path.exists():
        log.info("AbLang-2 (paired): キャッシュを検出、スキップ")
        return

    log.info("AbLang-2 (paired): モデル読み込み中 ...")
    model = _ablang2_load_model(device)

    res_embs: list[np.ndarray] = []

    for i in tqdm(range(0, len(vh_seqs), batch_size), desc="AbLang-2 paired"):
        vh_batch = vh_seqs[i:i + batch_size]
        vl_batch = vl_seqs[i:i + batch_size]
        # w_extra_tkns=True: [<, VH..., >, |, <, VL..., >]
        pairs = [[vh, vl] for vh, vl in zip(vh_batch, vl_batch)]
        res_embs.extend(_ablang2_extract_vh(model, pairs, vh_batch, device))

    np.save(seq_path, pool_seq(res_embs))
    np.save(cdr3_path, pool_cdr3(res_embs, cdr3_indices))
    log.info(f"AbLang-2 (paired): 保存完了 → {seq_path}, {cdr3_path}")


def run_ablang2_heavy(
    vh_seqs: list[str],
    cdr3_indices: list[tuple[int, int] | None],
    batch_size: int,
    device: str,
    out_dir: Path,
) -> None:
    """AbLang-2 (VH 単独入力) の埋め込みを生成する。
    
    VL を空文字列として渡すことで、訓練時フォーマット [<,VH...,>,|] を維持しつつ
    重鎖のみのコンテキストで残基ベクトルを取得する。
    """
    seq_path = out_dir / "ablang2_heavy_seq_vecs.npy"
    cdr3_path = out_dir / "ablang2_heavy_cdr3_vecs.npy"
    if seq_path.exists() and cdr3_path.exists():
        log.info("AbLang-2 (heavy-only): キャッシュを検出、スキップ")
        return

    log.info("AbLang-2 (heavy-only): モデル読み込み中 ...")
    model = _ablang2_load_model(device)

    res_embs: list[np.ndarray] = []

    for i in tqdm(range(0, len(vh_seqs), batch_size), desc="AbLang-2 heavy"):
        vh_batch = vh_seqs[i:i + batch_size]
        # VL を空文字列 → encode: f"<{vh}>|<{}>".replace("<>","") = "<{vh}>|"
        # トークン列: [<, VH残基..., >, |]
        pairs = [[vh, ""] for vh in vh_batch]
        res_embs.extend(_ablang2_extract_vh(model, pairs, vh_batch, device))

    np.save(seq_path, pool_seq(res_embs))
    np.save(cdr3_path, pool_cdr3(res_embs, cdr3_indices))
    log.info(f"AbLang-2 (heavy-only): 保存完了 → {seq_path}, {cdr3_path}")


# ──────────────────────────────────────────────
# ESM-2
# ──────────────────────────────────────────────

def run_esm2(
    vh_seqs: list[str],
    cdr3_indices: list[tuple[int, int] | None],
    batch_size: int,
    device: str,
    out_dir: Path,
) -> None:
    """ESM-2 (facebook/esm2_t33_650M_UR50D) の埋め込みを生成する。

    トークンレイアウト: [CLS(0), AA_0(1), ..., AA_{L-1}(L), EOS(L+1)]
    VH 残基の hidden state インデックス: 1 〜 1+len(seq)
    """
    seq_path = out_dir / "esm2_seq_vecs.npy"
    cdr3_path = out_dir / "esm2_cdr3_vecs.npy"
    if seq_path.exists() and cdr3_path.exists():
        log.info("ESM-2: キャッシュを検出、スキップ")
        return

    from transformers import EsmModel, AutoTokenizer

    MODEL_ID = "facebook/esm2_t33_650M_UR50D"
    log.info(f"ESM-2: モデル読み込み中 ({MODEL_ID}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = EsmModel.from_pretrained(MODEL_ID).to(device).eval()

    res_embs: list[np.ndarray] = []

    for i in tqdm(range(0, len(vh_seqs), batch_size), desc="ESM-2"):
        batch = vh_seqs[i:i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, add_special_tokens=True,
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        for j, seq in enumerate(batch):
            # index 0 = CLS, index 1..L = VH残基, index L+1 = EOS
            residue_embs = out.last_hidden_state[j, 1:1 + len(seq)].cpu().float().numpy()
            res_embs.append(residue_embs)

    np.save(seq_path, pool_seq(res_embs))
    np.save(cdr3_path, pool_cdr3(res_embs, cdr3_indices))
    log.info(f"ESM-2: 保存完了 → {seq_path}, {cdr3_path}")


# ──────────────────────────────────────────────
# ProGen-2
# ──────────────────────────────────────────────

def run_progen2(
    vh_seqs: list[str],
    cdr3_indices: list[tuple[int, int] | None],
    device: str,
    out_dir: Path,
) -> None:
    """ProGen-2 (hugohrban/progen2-base) の埋め込みを生成する。

    Causal LM のため個別処理（パディング不要）。
    トークンレイアウト: [BOS(0), AA_0(1), ..., AA_{L-1}(L), EOS(L+1)]
    VH 残基の hidden state インデックス: 1 〜 1+len(seq)
    """
    seq_path = out_dir / "progen2_seq_vecs.npy"
    cdr3_path = out_dir / "progen2_cdr3_vecs.npy"
    if seq_path.exists() and cdr3_path.exists():
        log.info("ProGen-2: キャッシュを検出、スキップ")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_ID = "hugohrban/progen2-base"
    log.info(f"ProGen-2: モデル読み込み中 ({MODEL_ID}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
    ).to(device).eval()

    res_embs: list[np.ndarray] = []

    for seq in tqdm(vh_seqs, desc="ProGen-2"):
        enc = tokenizer(seq, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[-1][0]  # (L_tok, D)
        # BOS をスキップして VH 残基のみ抽出
        residue_embs = hidden[1:1 + len(seq)].cpu().float().numpy()  # (L, D)
        res_embs.append(residue_embs)

    np.save(seq_path, pool_seq(res_embs))
    np.save(cdr3_path, pool_cdr3(res_embs, cdr3_indices))
    log.info(f"ProGen-2: 保存完了 → {seq_path}, {cdr3_path}")


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="5 モデルの埋め込みを生成して保存する")
    parser.add_argument("--datadir", type=str, default="data", help="データディレクトリ (default: data)")
    parser.add_argument("--batch", type=int, default=32, help="バッチサイズ (default: 32)")
    parser.add_argument(
        "--device", type=str, default="auto",
        help="使用デバイス: auto / cpu / cuda (default: auto)"
    )
    parser.add_argument(
        "--model", type=str, default="all",
        help="実行するモデル: all / antiberty / ablang / ablang2 / ablang2heavy / esm2 / progen2 (default: all)"
    )
    args = parser.parse_args()

    data_dir = Path(args.datadir)
    csv_path = data_dir / "sequences_processed.csv"
    if not csv_path.exists():
        log.error(f"{csv_path} が見つかりません。先に 01_download_data.py を実行してください。")
        sys.exit(1)

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    log.info(f"使用デバイス: {device}")

    # データ読み込み
    df = pd.read_csv(csv_path)
    log.info(f"読み込んだ配列数: {len(df)}")

    vh_seqs = df["vh_seq"].tolist()
    vl_seqs = df["vl_seq"].tolist()

    # cdr3_indices は CSV に文字列として保存されているのでパース
    def parse_idx(val):
        if pd.isna(val) or val is None:
            return None
        if isinstance(val, tuple):
            return val
        try:
            result = ast.literal_eval(str(val))
            return tuple(result) if result is not None else None
        except Exception:
            return None

    cdr3_indices = [parse_idx(v) for v in df["cdr3_indices"]]

    # 埋め込み生成
    models_to_run = set(
        ["antiberty", "ablang", "ablang2", "ablang2heavy", "esm2", "progen2"]
        if args.model == "all"
        else [args.model]
    )

    if "antiberty" in models_to_run:
        run_antiberty(vh_seqs, cdr3_indices, args.batch, data_dir)

    if "ablang" in models_to_run:
        run_ablang(vh_seqs, cdr3_indices, args.batch, data_dir)

    # AbLang-2 系はペア入力のためバッチサイズを小さめに
    batch2 = min(args.batch, 16)

    if "ablang2" in models_to_run:
        run_ablang2(vh_seqs, vl_seqs, cdr3_indices, batch2, device, data_dir)

    if "ablang2heavy" in models_to_run:
        run_ablang2_heavy(vh_seqs, cdr3_indices, batch2, device, data_dir)

    if "esm2" in models_to_run:
        run_esm2(vh_seqs, cdr3_indices, args.batch, device, data_dir)

    if "progen2" in models_to_run:
        run_progen2(vh_seqs, cdr3_indices, device, data_dir)

    log.info("全モデルの埋め込み生成完了")


if __name__ == "__main__":
    main()

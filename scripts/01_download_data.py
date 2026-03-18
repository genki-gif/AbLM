"""
01_download_data.py
OAS Paired からヒト抗体配列をダウンロードし、前処理・CDR3特定を行って保存する。

出力:
    data/sequences_processed.csv   前処理済み配列（vh_seq, vl_seq, v_family, cdr3_indices 等）

使用方法:
    mamba run -n ablm python scripts/01_download_data.py [--n 1000] [--outdir data]
"""

import argparse
import gzip
import io
import logging
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from abnumber import Chain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# OAS ダウンロード
# ──────────────────────────────────────────────

# ngsdb の paired インデックス URL
OAS_NGSDB_PAIRED = "https://opig.stats.ox.ac.uk/webapps/ngsdb/paired/"
RANDOM_STATE = 42


def _list_hrefs(url: str) -> list[str]:
    """Apache style インデックスページからリンク名（末尾'/'付き）を返す。"""
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    import re
    return re.findall(r'href="([^"?/][^"]*)"', r.text)


def _read_unit_gz(url: str) -> pd.DataFrame | None:
    """
    OAS paired .csv.gz を読み込む。
    1 行目 = JSON メタデータ → header=1 でスキップ。
    """
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        content = gzip.decompress(r.content)
        return pd.read_csv(io.StringIO(content.decode("utf-8")), header=1)
    except Exception as e:
        log.warning(f"スキップ ({url}): {e}")
        return None


def _get_species_from_gz(url: str) -> str | None:
    """
    .csv.gz の先頭行 (メタデータ JSON) から Species を読み取る。
    ダウンロード量を抑えるため Range ヘッダで先頭 4KB のみ取得する。
    """
    try:
        r = requests.get(url, headers={"Range": "bytes=0-4095"}, timeout=30)
        # gunzip の途中切れでも先頭行だけ取れれば十分
        import zlib
        d = zlib.decompressobj(zlib.MAX_WBITS | 16)
        text = d.decompress(r.content).decode("utf-8", errors="ignore")
        first_line = text.splitlines()[0].strip().strip('"').replace('""', '"')
        import json
        meta = json.loads(first_line)
        return meta.get("Species", "").lower()
    except Exception:
        return None


def download_oas_paired(n_target: int, species: str = "human") -> pd.DataFrame:
    """
    ngsdb/paired/ を巡回してヒト配列を集める。
    各スタディの先頭ファイルで種を確認し、human のみダウンロードする。
    """
    log.info("OAS ngsdb/paired/ からスタディ一覧を取得中 ...")
    studies = [s for s in _list_hrefs(OAS_NGSDB_PAIRED) if s.endswith("/")]
    log.info(f"スタディ数: {len(studies)}")

    dfs: list[pd.DataFrame] = []
    collected = 0

    for study in tqdm(studies, desc="スタディ巡回"):
        if collected >= n_target:
            break
        csv_index_url = f"{OAS_NGSDB_PAIRED}{study}csv/"
        try:
            files = [f for f in _list_hrefs(csv_index_url) if f.endswith(".csv.gz")]
        except Exception:
            continue
        if not files:
            continue

        # 先頭ファイルで species を確認
        sample_url = f"{csv_index_url}{files[0]}"
        sp = _get_species_from_gz(sample_url)
        if sp != species:
            log.info(f"  {study.rstrip('/')}: species={sp} → スキップ")
            continue
        log.info(f"  {study.rstrip('/')}: species={sp} → {len(files)} ファイル")

        for fname in files:
            if collected >= n_target:
                break
            url = f"{csv_index_url}{fname}"
            df_unit = _read_unit_gz(url)
            if df_unit is not None and len(df_unit) > 0:
                dfs.append(df_unit)
                collected += len(df_unit)
                log.info(f"    {fname}: {len(df_unit)} 配列 (累計 {collected})")

    if not dfs:
        raise RuntimeError(
            "ngsdb からデータを取得できませんでした。\n"
            "--local オプションを使いローカルの .csv.gz を読み込んでください。"
        )

    df = pd.concat(dfs, ignore_index=True)
    log.info(f"ダウンロード完了: {len(df)} 配列")
    return df


def load_local_csvgz(data_dir: Path) -> pd.DataFrame:
    """data/ フォルダ内の *.csv.gz を結合して読み込む（手動 DL 用）。"""
    files = list(data_dir.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(
            f"{data_dir} に .csv.gz ファイルが見つかりません。"
            "OAS から wget スクリプトを取得して実行してください。"
        )
    log.info(f"{len(files)} 件のファイルを読み込み中 ...")
    dfs = []
    for f in tqdm(files, desc="ローカルファイル読み込み"):
        try:
            dfs.append(pd.read_csv(f, header=1, compression="gzip"))
        except Exception as e:
            log.warning(f"スキップ: {f.name}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    log.info(f"合計 {len(df)} 配列を読み込み")
    return df


# ──────────────────────────────────────────────
# 前処理
# ──────────────────────────────────────────────

def preprocess(df: pd.DataFrame, n: int = 1000) -> pd.DataFrame:
    """
    カラム名の標準化・フィルタリング・V遺伝子ファミリー抽出・均等サンプリング。
    """
    # カラム名を標準化
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "sequence_alignment_aa_heavy":
            rename[c] = "vh_seq"
        elif cl == "sequence_alignment_aa_light":
            rename[c] = "vl_seq"
        elif cl == "v_call_heavy":
            rename[c] = "v_call_heavy"
        elif cl == "cdr3_aa_heavy":
            rename[c] = "cdr3_aa_heavy"
    df = df.rename(columns=rename)

    required = ["vh_seq", "vl_seq", "v_call_heavy"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"必要なカラムが見つかりません: {missing}\n利用可能: {df.columns.tolist()}"
        )

    df = df.dropna(subset=required)
    df = df[df["vh_seq"].str.len() >= 80]
    df = df[df["vl_seq"].str.len() >= 80]
    df = df[~df["vh_seq"].str.contains(r"[X*\-.]", regex=True, na=False)]
    df = df[~df["vl_seq"].str.contains(r"[X*\-.]", regex=True, na=False)]

    # V遺伝子ファミリー
    df["v_family"] = df["v_call_heavy"].str.extract(r"(IGHV\d+)")
    df = df.dropna(subset=["v_family"])
    major = [f"IGHV{i}" for i in range(1, 8)]
    df = df[df["v_family"].isin(major)]

    # 均等サンプリング
    n_families = df["v_family"].nunique()
    n_per = n // n_families
    df = (
        df.groupby("v_family", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), n_per), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )

    log.info(f"前処理後配列数: {len(df)}")
    log.info("V遺伝子ファミリー分布:\n" + df["v_family"].value_counts().sort_index().to_string())
    return df


# ──────────────────────────────────────────────
# CDR3 特定
# ──────────────────────────────────────────────

def get_cdr3_indices_from_oas(vh_seq: str, cdr3_aa: str | None) -> tuple[int, int] | None:
    """OAS の cdr3_aa_heavy カラムを使って VH 中の CDR3 位置を特定する。"""
    if not (cdr3_aa and isinstance(cdr3_aa, str) and len(cdr3_aa) >= 3):
        return None
    # OAS の CDR3 はジャンクション境界残基を含む場合があるため、内側 (両端 2 残基除く) で検索
    core = cdr3_aa[2:-1] if len(cdr3_aa) > 5 else cdr3_aa
    idx = vh_seq.find(core)
    if idx < 0:
        return None
    start = max(0, idx - 2)
    end = start + len(cdr3_aa)
    if end > len(vh_seq):
        return None
    return start, end


def get_cdr3_indices_abnumber(vh_seq: str) -> tuple[int, int] | None:
    """AbNumber を使って CDR3 位置を特定する（フォールバック）。"""
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


def add_cdr3_info(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame に cdr3_indices / cdr3_seq_extracted / cdr3_valid カラムを追加する。"""
    has_col = "cdr3_aa_heavy" in df.columns
    indices, seqs, valid = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="CDR3 特定"):
        vh = row["vh_seq"]
        cdr3_aa = row.get("cdr3_aa_heavy") if has_col else None

        idx = get_cdr3_indices_from_oas(vh, cdr3_aa)
        if idx is None:
            idx = get_cdr3_indices_abnumber(vh)

        indices.append(idx)
        if idx is not None:
            seqs.append(vh[idx[0]:idx[1]])
            valid.append(True)
        else:
            seqs.append(None)
            valid.append(False)

    df = df.copy()
    df["cdr3_indices"] = indices
    df["cdr3_seq_extracted"] = seqs
    df["cdr3_valid"] = valid

    n_valid = sum(valid)
    log.info(
        f"CDR3 特定完了: {n_valid}/{len(df)} ({100 * n_valid / len(df):.1f}%)"
    )
    cdr3_lens = df.loc[df["cdr3_valid"], "cdr3_seq_extracted"].str.len()
    log.info(
        f"CDR3 長: min={cdr3_lens.min()}, max={cdr3_lens.max()}, "
        f"median={cdr3_lens.median():.0f}"
    )
    return df


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="OAS データのダウンロード・前処理・CDR3 特定")
    parser.add_argument("--n", type=int, default=1000, help="解析に使用する配列数 (default: 1000)")
    parser.add_argument("--outdir", type=str, default="data", help="出力ディレクトリ (default: data)")
    parser.add_argument(
        "--local", action="store_true",
        help="API ダウンロードをスキップし、outdir 内の .csv.gz を読み込む"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="既存の出力ファイルを上書きする"
    )
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sequences_processed.csv"

    if out_path.exists() and not args.force:
        log.info(f"既存のファイルを検出: {out_path} (スキップ。--force で上書き可)")
        return

    # ダウンロード
    if args.local:
        df_raw = load_local_csvgz(out_dir)
    else:
        try:
            df_raw = download_oas_paired(n_target=args.n * 5)
        except Exception as e:
            log.warning(f"API 失敗: {e}\nローカルファイルにフォールバック ...")
            df_raw = load_local_csvgz(out_dir)

    # 前処理
    df = preprocess(df_raw, n=args.n)

    # CDR3 特定
    df = add_cdr3_info(df)

    # CDR3 が特定できた行のみ保存
    df_valid = df[df["cdr3_valid"]].reset_index(drop=True)
    log.info(f"最終配列数 (CDR3 有効): {len(df_valid)}")

    df_valid.to_csv(out_path, index=False)
    log.info(f"保存完了: {out_path}")


if __name__ == "__main__":
    main()

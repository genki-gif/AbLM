"""
05_therapeutic_data.py
TheraSAbDab から治療用抗体配列を取得し、Pgen/PLL スコアを計算する。

出力 (data/therapeutic/ フォルダ):
    sequences_therapeutic.csv          前処理済み配列
    pgen.npy, pgen_vj.npy             OLGA Pgen
    *_pll.npy, *_cdr3_pll.npy         各モデル PLL
    pgen_pll_scores_therapeutic.csv    全スコアまとめ CSV

使用方法:
    KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/05_therapeutic_data.py
    KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/05_therapeutic_data.py --batch 32
    KMP_DUPLICATE_LIB_OK=TRUE mamba run -n ablm python scripts/05_therapeutic_data.py --source fallback
"""

import argparse
import ast
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm

from abnumber import Chain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# TheraSAbDab からのデータ取得
# ──────────────────────────────────────────────

THERASABDAB_URL = (
    "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/"
    "therasabdab/search/?all=true&format=tsv"
)


def download_therasabdab() -> pd.DataFrame:
    """TheraSAbDab TSV をダウンロードして DataFrame を返す。"""
    log.info(f"TheraSAbDab からデータ取得中: {THERASABDAB_URL}")
    r = requests.get(THERASABDAB_URL, timeout=300)
    r.raise_for_status()
    import io

    df = pd.read_csv(io.StringIO(r.text), sep="\t")
    log.info(f"TheraSAbDab: {len(df)} エントリ取得")
    return df


# 主要治療用抗体のフォールバックデータ（VH 配列のみ, VL は空）
FALLBACK_THERAPEUTICS = [
    {
        "drug_name": "Trastuzumab",
        "vh_seq": "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS",
    },
    {
        "drug_name": "Pembrolizumab",
        "vh_seq": "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS",
    },
    {
        "drug_name": "Adalimumab",
        "vh_seq": "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
    },
    {
        "drug_name": "Nivolumab",
        "vh_seq": "QVQLVESGGGVVQPGRSLRLDCKASGITFSNSGMHWVRQAPGKGLEWVAVIWYDGSKRYYADSVKGRFTISRDNSKNTLFLQMNSLRAEDTAVYYCATNDDYWGQGTLVTVSS",
    },
    {
        "drug_name": "Rituximab",
        "vh_seq": "QVQLQQPGAELVKPGASVKMSCKASGYTFTSYNMHWVKQTPGRGLEWIGAIYPGNGDTSYNQKFKGKATLTADKSSSTAYMQLSSLTSEDSAVYYCARSTYYGGDWYFNVWGAGTTVTVSA",
    },
    {
        "drug_name": "Bevacizumab",
        "vh_seq": "EVQLVESGGGLVQPGGSLRLSCAASGYTFTNYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAVYYCAKYPHYYGSSHWYFDVWGQGTLVTVSS",
    },
    {
        "drug_name": "Cetuximab",
        "vh_seq": "QVQLKQSGPGLVQPSQSLSITCTVSGFSLTNYGVHWVRQSPGKGLEWLGVIWSGGNTDYNTPFTSRLSINKDNSKSQVFFKMNSLQSNDTAIYYCARALTYYDYEFAYWGQGTLVTVSA",
    },
    {
        "drug_name": "Infliximab",
        "vh_seq": "EVKLEESGGGLVQPGGSMKLSCVASGFIFSNHWMNWVRQSPEKGLEWVAEIRSKSINSATHYAESVKGRFTISRDDSKSSAYLQMNNLRTEDTGIYYCSRNYYGSTYDYWGQGTTLTVSS",
    },
    {
        "drug_name": "Ipilimumab",
        "vh_seq": "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYTMHWVRQAPGKGLEWVTFISYDGNNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAIYYCARTGWLGPFDYWGQGTLVTVSS",
    },
    {
        "drug_name": "Atezolizumab",
        "vh_seq": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDSWIHWVRQAPGKGLEWVAWISPYGGSTYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCARRHWPGGFDYWGQGTLVTVSS",
    },
    {
        "drug_name": "Durvalumab",
        "vh_seq": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSRYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAREGGWFGELAFDYWGQGTLVTVSS",
    },
    {
        "drug_name": "Avelumab",
        "vh_seq": "EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYIMMWVRQAPGKGLEWVSSIYPSGGITFYADTVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARIKLGTVTTVDYWGQGTLVTVSS",
    },
    {
        "drug_name": "Palivizumab",
        "vh_seq": "QITLKESGPGIVQPSQPFRLTCTFSGFSLSTSGMGVSWIRQPSGKGLEWLAHIYWDDDKRYNPSLKSRLTISKDTSKNQVSLKITSVTAADTAVYYCARLKFYTGGQGTHFDYWGQGTLVTVSS",
    },
    {
        "drug_name": "Omalizumab",
        "vh_seq": "EVQLVESGGGLVQPGRSLRLSCAASGFTFSDYAMSWVRQAPGKGLEWVSYISSGSSTIYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARLRHEDAYYGSGSYYPEDYWGQGTLVTVSS",
    },
    {
        "drug_name": "Secukinumab",
        "vh_seq": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSNYWMNWVRQAPGKGLEWVGAINPDGGSTYYVDSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYCARDRYGFLHYFDIWGQGTLVTVSS",
    },
    {
        "drug_name": "Ustekinumab",
        "vh_seq": "EVQLVESGGGLVKPGGSLRLSCEASGFTFSRYWMSWVRQAPGKGLEWIGEINPDSSTINYAPSFKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYCARPDGNYWYFDLWGQGTLVTVSS",
    },
    {
        "drug_name": "Tocilizumab",
        "vh_seq": "QVQLQESGPGLVKPSETLSLTCAVSGYSISSSNWWGWIRQPPGKGLEWIGEINHSGSTNYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARDRGIAAGGNYYYYGMDVWGQGTTVTVSS",
    },
    {
        "drug_name": "Dupilumab",
        "vh_seq": "EVQLVESGGGLEQPGGSLRLSCAGSGFTFRDYAMTWVRQAPGKGLEWVSSISGSGGNTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRLSITIRPRYYGLDVWGQGTTVTVSS",
    },
    {
        "drug_name": "Pertuzumab",
        "vh_seq": "EVQLVESGGGLVQPGGSLRLSCAASGFTISDYWIHWVRQAPGKGLEWVAGITPAGGYTYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRSRYNDYYDTMDYWGQGTLVTVSS",
    },
    {
        "drug_name": "Natalizumab",
        "vh_seq": "QVQLVQSGAEVKKPGSSVKVSCKASGYTITDSNIHWVRQAPGQGLEWIGDIIPILGIANYAQKFQGRVTITADKSTSTVYMELSSLRSEDTAVYYCARGIILDYGDYRLDYWGQGTLVTVSS",
    },
]


def get_fallback_data() -> pd.DataFrame:
    """ハードコードの治療用抗体データを返す。"""
    log.info("フォールバックデータ（主要 20 抗体）を使用")
    return pd.DataFrame(FALLBACK_THERAPEUTICS)


# ──────────────────────────────────────────────
# TheraSAbDab → パイプライン形式に変換
# ──────────────────────────────────────────────

def _find_vh_col(df: pd.DataFrame) -> str | None:
    """TheraSAbDab の VH 配列カラム名を検出する。"""
    candidates = ["VH", "Hchain", "Heavy Sequence", "vh_seq", "VH_aa"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "heavy" in c.lower() or "vh" in c.lower():
            if "seq" in c.lower() or "chain" in c.lower() or "aa" in c.lower():
                return c
    return None


def _find_vl_col(df: pd.DataFrame) -> str | None:
    """TheraSAbDab の VL 配列カラム名を検出する。"""
    candidates = ["VL", "Lchain", "Light Sequence", "vl_seq", "VL_aa"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "light" in c.lower() or "vl" in c.lower():
            if "seq" in c.lower() or "chain" in c.lower() or "aa" in c.lower():
                return c
    return None


def _find_name_col(df: pd.DataFrame) -> str | None:
    """TheraSAbDab の薬品名カラムを検出する。"""
    candidates = [
        "Therapeutic", "Name", "Drug", "INN",
        "therapeutic", "name", "drug_name",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def process_therasabdab(df_raw: pd.DataFrame) -> pd.DataFrame:
    """TheraSAbDab DataFrame をパイプライン形式に変換する。"""
    vh_col = _find_vh_col(df_raw)
    vl_col = _find_vl_col(df_raw)
    name_col = _find_name_col(df_raw)

    if vh_col is None:
        raise ValueError(
            f"VH 配列カラムが見つかりません。利用可能: {df_raw.columns.tolist()}"
        )

    log.info(f"カラムマッピング: VH={vh_col}, VL={vl_col}, Name={name_col}")

    records: list[dict] = []
    for _, row in df_raw.iterrows():
        vh = str(row[vh_col]) if pd.notna(row[vh_col]) else ""
        vl = str(row[vl_col]) if vl_col and pd.notna(row.get(vl_col)) else ""
        name = str(row[name_col]) if name_col and pd.notna(row.get(name_col)) else ""

        # フィルタ: VH が有効か
        if len(vh) < 80:
            continue
        if re.search(r"[X*\-.]", vh):
            continue

        records.append({
            "drug_name": name,
            "vh_seq": vh,
            "vl_seq": vl,
        })

    df = pd.DataFrame(records)
    # 同一薬品名で重複する場合は最初のものを使用
    if "drug_name" in df.columns and len(df) > 0:
        df = df.drop_duplicates(subset="drug_name", keep="first")
    log.info(f"フィルタ後: {len(df)} 配列")
    return df


# ──────────────────────────────────────────────
# V/J アノテーション + CDR3 特定
# ──────────────────────────────────────────────

def annotate_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """AbNumber で V/J 遺伝子アノテーションと CDR3 位置を特定する。"""
    v_calls = []
    j_calls = []
    v_families = []
    cdr3_aa_list = []
    cdr3_indices_list = []
    valid_rows = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="V/J アノテーション"):
        vh = row["vh_seq"]
        try:
            chain = Chain(vh, scheme="imgt", assign_germline=True)
        except Exception as e:
            log.warning(f"AbNumber 失敗 ({row.get('drug_name', i)}): {e}")
            continue

        v_gene = chain.v_gene or ""
        j_gene = chain.j_gene or ""
        cdr3 = chain.cdr3_seq or ""

        # V ファミリー抽出
        m = re.search(r"(IGHV\d+)", v_gene)
        v_fam = m.group(1) if m else ""

        # CDR3 インデックス特定
        cdr3_idx = None
        if cdr3 and len(cdr3) >= 3:
            idx = vh.find(cdr3)
            if idx >= 0:
                cdr3_idx = (idx, idx + len(cdr3))

        # cdr3_aa_heavy (OAS 形式: 保存 C/W を除去)
        cdr3_aa = ""
        if cdr3 and len(cdr3) >= 2:
            # Chain.cdr3_seq は IMGT 定義で保存残基を含む場合がある
            # OAS 形式では先頭 C と末尾 W/F を除去
            cdr3_aa = cdr3
            if cdr3_aa.startswith("C"):
                cdr3_aa = cdr3_aa[1:]
            if cdr3_aa and cdr3_aa[-1] in "WF":
                cdr3_aa = cdr3_aa[:-1]

        v_calls.append(v_gene)
        j_calls.append(j_gene)
        v_families.append(v_fam)
        cdr3_aa_list.append(cdr3_aa)
        cdr3_indices_list.append(cdr3_idx)
        valid_rows.append(i)

    df_valid = df.loc[valid_rows].copy()
    df_valid["v_call_heavy"] = v_calls
    df_valid["j_call_heavy"] = j_calls
    df_valid["v_family"] = v_families
    df_valid["cdr3_aa_heavy"] = cdr3_aa_list
    df_valid["cdr3_indices"] = cdr3_indices_list

    log.info(f"アノテーション完了: {len(df_valid)}/{len(df)} 配列")
    return df_valid.reset_index(drop=True)


# ──────────────────────────────────────────────
# スコア計算（03_compute_pgen_pll.py の関数を再利用）
# ──────────────────────────────────────────────

def compute_scores(
    df: pd.DataFrame,
    out_dir: Path,
    batch: int,
    device: str,
    models_to_run: set[str],
) -> None:
    """03_compute_pgen_pll.py の各 run_* 関数を呼び出してスコアを計算する。"""
    # 03_compute_pgen_pll.py をインポート
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "compute_pgen_pll",
        Path(__file__).parent / "03_compute_pgen_pll.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    vh_seqs = df["vh_seq"].tolist()
    vl_seqs = df["vl_seq"].tolist() if "vl_seq" in df.columns else [""] * len(df)

    def parse_idx(val):
        if pd.isna(val) or val is None:
            return None
        try:
            result = ast.literal_eval(str(val)) if isinstance(val, str) else val
            return tuple(result) if result is not None else None
        except Exception:
            return None

    cdr3_indices = [parse_idx(v) for v in df["cdr3_indices"]]

    inner_batch2 = min(batch, 16)

    if "pgen" in models_to_run:
        mod.run_pgen(df, out_dir)

    if "antiberty" in models_to_run:
        mod.run_antiberty_pll(vh_seqs, cdr3_indices, batch, device, out_dir)

    if "ablang" in models_to_run:
        mod.run_ablang_pll(vh_seqs, cdr3_indices, batch, device, out_dir)

    if "ablang2" in models_to_run:
        mod.run_ablang2_pll(vh_seqs, vl_seqs, cdr3_indices, inner_batch2, device, out_dir)

    if "ablang2heavy" in models_to_run:
        mod.run_ablang2_heavy_pll(vh_seqs, cdr3_indices, inner_batch2, device, out_dir)

    if "esm2" in models_to_run:
        mod.run_esm2_pll(vh_seqs, cdr3_indices, batch, device, out_dir)

    if "progen2" in models_to_run:
        mod.run_progen2_pll(vh_seqs, cdr3_indices, device, out_dir)

    # CSV 構築（build_csv + drug_name 追加）
    mod.build_csv(df, out_dir)

    # drug_name カラムを追加
    csv_path = out_dir / "pgen_pll_scores.csv"
    if csv_path.exists():
        df_scores = pd.read_csv(csv_path)
        df_scores.insert(0, "drug_name", df["drug_name"].values[: len(df_scores)])
        out_csv = out_dir / "pgen_pll_scores_therapeutic.csv"
        df_scores.to_csv(out_csv, index=False)
        log.info(f"治療用抗体スコア CSV 保存完了 → {out_csv}")


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="治療用抗体データの取得・前処理・Pgen/PLL スコア計算"
    )
    parser.add_argument(
        "--outdir", type=str, default="data/therapeutic",
        help="出力ディレクトリ (default: data/therapeutic)",
    )
    parser.add_argument(
        "--batch", type=int, default=32, help="内部バッチサイズ (default: 32)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="使用デバイス: auto / cpu / cuda (default: auto)",
    )
    parser.add_argument(
        "--model", type=str, default="all",
        help=(
            "実行するモデル: "
            "all / pgen / antiberty / ablang / ablang2 / ablang2heavy / esm2 / progen2 "
            "(default: all)"
        ),
    )
    parser.add_argument(
        "--source", type=str, default="therasabdab",
        choices=["therasabdab", "fallback"],
        help="データソース: therasabdab (TheraSAbDab API) / fallback (ハードコード 20 抗体)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="既存の出力ファイルを上書きする",
    )
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_path = out_dir / "sequences_therapeutic.csv"

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    log.info(f"使用デバイス: {device}")

    # ステップ 1: 配列取得・前処理
    if seq_path.exists() and not args.force:
        log.info(f"既存のファイルを検出: {seq_path} (スキップ。--force で上書き可)")
        df = pd.read_csv(seq_path)
    else:
        if args.source == "therasabdab":
            try:
                df_raw = download_therasabdab()
                df = process_therasabdab(df_raw)
            except Exception as e:
                log.warning(f"TheraSAbDab 取得失敗: {e}\nフォールバックデータを使用")
                df = get_fallback_data()
        else:
            df = get_fallback_data()

        # V/J アノテーション + CDR3 特定
        df = annotate_sequences(df)
        df.to_csv(seq_path, index=False)
        log.info(f"配列データ保存完了 → {seq_path} ({len(df)} 配列)")

    log.info(f"治療用抗体配列数: {len(df)}")

    # ステップ 2: スコア計算
    models_to_run = set(
        ["pgen", "antiberty", "ablang", "ablang2", "ablang2heavy", "esm2", "progen2"]
        if args.model == "all"
        else [args.model]
    )

    compute_scores(df, out_dir, args.batch, device, models_to_run)
    log.info("治療用抗体パイプライン完了")


if __name__ == "__main__":
    main()

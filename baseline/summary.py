import argparse
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


# ====================================================================
# Đọc kết quả
# ====================================================================
def load_heuristic_results(result_root: Path) -> list:
    """Đọc kết quả các thuật toán heuristic: result/<algo>/<group>/<file>.json"""
    rows = []
    if not result_root.exists():
        return rows
    for algo_dir in sorted(p for p in result_root.iterdir() if p.is_dir()):
        for f in sorted(algo_dir.rglob("*.json")):
            with open(f, encoding="utf-8") as fp:
                d = json.load(fp)
            rows.append({
                "algo": algo_dir.name.upper(),
                "group": f.parent.relative_to(algo_dir).as_posix(),
                "instance": f.stem,
                "objective": d.get("best_profit"),
            })
    return rows


def load_milp_results(milp_root: Path) -> list:
    """Đọc kết quả MILP: milp/result/f1/<group>/<file>.json"""
    rows = []
    if not milp_root.exists():
        return rows
    for f in sorted(milp_root.rglob("*.json")):
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        rows.append({
            "algo": "MILP",
            "group": f.parent.relative_to(milp_root).as_posix(),
            "instance": f.stem,
            "objective": d.get("objective"),
        })
    return rows


def instance_type(name: str) -> str:
    """S042_N10_RC_R3 -> RC"""
    parts = name.split("_")
    return parts[2] if len(parts) > 2 else ""


# ====================================================================
# Thống kê
# ====================================================================
def build_summary(df: pd.DataFrame):
    df = df.copy()
    df["type"] = df["instance"].map(instance_type)

    # Thuật toán đạt objective lớn nhất trên mỗi instance
    best = df.groupby(["group", "instance"])["objective"].transform("max")
    df["is_best"] = (df["objective"] >= best - 1e-6).astype(int)

    # Bảng theo từng bộ: mỗi hàng là một instance, mỗi cột là một thuật toán
    per_group = {}
    for group, g in df.groupby("group"):
        table = g.pivot_table(index="instance", columns="algo", values="objective", aggfunc="first")
        table = table[sorted(table.columns)]
        table.columns.name = None
        per_group[group] = table

    agg = dict(
        n=("objective", "size"),
        obj_mean=("objective", "mean"),
        obj_std=("objective", "std"),
        obj_min=("objective", "min"),
        obj_max=("objective", "max"),
        n_best=("is_best", "sum"),
    )
    by_group = df.groupby(["group", "algo"]).agg(**agg)
    by_type = df.groupby(["group", "type", "algo"]).agg(**agg)

    # Tổng quát: chỉ so sánh trên các instance mà mọi thuật toán đều có kết quả
    algos = df["algo"].unique()
    counts = df.groupby(["group", "instance"])["algo"].nunique()
    common = counts[counts == len(algos)].index
    df_common = df.set_index(["group", "instance"]).loc[common].reset_index()
    overall = df_common.groupby("algo").agg(**agg) if len(df_common) else pd.DataFrame()

    return df, per_group, by_group, by_type, overall


def print_section(title: str, table: pd.DataFrame):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    if table.empty:
        print("(không có dữ liệu)")
    else:
        print(table.to_string(float_format=lambda x: f"{x:.2f}"))


def main():
    parser = argparse.ArgumentParser(description="Thống kê objective của các thuật toán TD-TOP")
    parser.add_argument("--result-root", type=str, default=str(BASE_DIR / "result"))
    parser.add_argument("--milp-root", type=str, default=str(BASE_DIR.parent / "milp" / "result" / "f1"))
    parser.add_argument("--no-milp", action="store_true", help="Bỏ qua kết quả MILP")
    parser.add_argument("--group", type=str, default=None, help="Chỉ thống kê nhóm này, vd S/N10")
    parser.add_argument("--algos", type=str, nargs="*", default=None, help="Chỉ thống kê các thuật toán này, vd GA LNS")
    parser.add_argument("--out-dir", type=str, default=str(BASE_DIR / "result"), help="Thư mục lưu CSV")
    parser.add_argument("--no-instances", action="store_true", help="Không in bảng theo từng instance")
    args = parser.parse_args()

    rows = load_heuristic_results(Path(args.result_root))
    if not args.no_milp:
        rows += load_milp_results(Path(args.milp_root))
    if not rows:
        print("Không tìm thấy kết quả nào.")
        return

    df = pd.DataFrame(rows)
    if args.group:
        df = df[df["group"] == args.group]
    if args.algos:
        df = df[df["algo"].isin([a.upper() for a in args.algos])]
    if df.empty:
        print("Không có kết quả phù hợp bộ lọc.")
        return

    df, per_group, by_group, by_type, overall = build_summary(df)

    if not args.no_instances:
        for group, table in per_group.items():
            print_section(f"OBJECTIVE BỘ {group}", table)
    print_section("THỐNG KÊ THEO NHÓM", by_group)
    print_section("THỐNG KÊ THEO NHÓM VÀ LOẠI INSTANCE (C / R / RC)", by_type)
    print_section("TỔNG QUÁT", overall)

    out_dir = Path(args.out_dir)
    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for group, table in per_group.items():
        table.to_csv(summary_dir / f"{group.replace('/', '_')}.csv")
    df.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nĐã lưu CSV từng bộ vào {summary_dir}")


if __name__ == "__main__":
    main()

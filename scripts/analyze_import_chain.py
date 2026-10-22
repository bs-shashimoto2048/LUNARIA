"""ACE-Step の import 連鎖を AST で静的に解析する（コードは実行しない）。

acestep.handler から、モジュール直下の import（eager = import した瞬間に実行される）をたどり、
  - 到達する acestep 内のモジュール数
  - 各サードパーティのパッケージを最初に import するモジュール（= 連鎖の経路）
  - 関数内の import（lazy = 呼ばれるまで読み込まれない）で、eager 化している候補
  - import 時（モジュール直下）に実行される関数呼び出し（CUDA・モデル・大きな確保の疑い）
を出力する。使い方: python -I scripts/analyze_import_chain.py [起点モジュール] [--json]
"""
import ast
import json
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = ROOT / "engine" / "ACE-Step-1.5"
STDLIB = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()
import re
RISKY_RE = re.compile(r"(\.to\(|cuda|from_pretrained|load_state_dict|torch\.(empty|zeros|randn)|\.init\()")


def module_path(name):
    parts = name.split(".")
    p = PKG_ROOT.joinpath(*parts)
    if (p / "__init__.py").exists():
        return p / "__init__.py"
    if p.with_suffix(".py").exists():
        return p.with_suffix(".py")
    return None


def resolve_from(cur, node):
    """相対 import を絶対名へ。cur は現在のモジュール名、パッケージかどうかはファイルで判定。"""
    if node.level == 0:
        return node.module or ""
    is_pkg = module_path(cur) and module_path(cur).name == "__init__.py"
    base = cur.split(".") if is_pkg else cur.split(".")[:-1]
    base = base[: len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def analyze(start):
    seen, order, third, lazy, import_time_calls = set(), [], OrderedDict(), OrderedDict(), OrderedDict()
    def parents(mod):
        """a.b.c を import すると、a と a.b の __init__ も先に実行される（暗黙の連鎖）。"""
        parts = mod.split(".")
        return [".".join(parts[:i]) for i in range(1, len(parts))]

    stack = [(p, "(親パッケージ)") for p in parents(start)] + [(start, None)]
    while stack:
        name, via = stack.pop(0)
        if name in seen:
            continue
        path = module_path(name)
        if path is None:
            continue
        seen.add(name)
        order.append((name, via))
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        # モジュール直下（if/try の中は含める。def/class の中は含めない）
        def walk(body, eager):
            for n in body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    walk(n.body, False)
                elif isinstance(n, ast.ClassDef):
                    walk(n.body, eager)
                elif isinstance(n, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                    for attr in ("body", "orelse", "finalbody", "handlers"):
                        sub = getattr(n, attr, None)
                        if sub:
                            for s in sub:
                                walk(s.body if isinstance(s, ast.ExceptHandler) else [s], eager)
                elif isinstance(n, (ast.Import, ast.ImportFrom)):
                    mods = ([a.name for a in n.names] if isinstance(n, ast.Import)
                            else [resolve_from(name, n)] + [resolve_from(name, n) + "." + a.name for a in n.names])
                    for m in mods:
                        top = m.split(".")[0]
                        if top == "acestep":
                            if eager and module_path(m):
                                for p in parents(m):
                                    stack.append((p, name))
                                stack.append((m, name))
                        elif top and top not in STDLIB:
                            (third if eager else lazy).setdefault(top, []).append(name)
                elif eager and isinstance(n, (ast.Expr, ast.Assign)) and isinstance(getattr(n, "value", None), ast.Call):
                    src = ast.unparse(n.value)[:120]
                    if RISKY_RE.search(src):
                        import_time_calls.setdefault(name, []).append(src)
        walk(tree.body, True)
    return order, third, lazy, import_time_calls


def main():
    start = next((a for a in sys.argv[1:] if not a.startswith("--")), "acestep.handler")
    order, third, lazy, calls = analyze(start)
    res = {"start": start, "eager_acestep_modules": len(order), "modules": [m for m, _ in order],
           "third_party_eager": {k: {"first_importer": v[0], "importers": len(set(v))} for k, v in third.items()},
           "third_party_lazy_only": sorted(set(lazy) - set(third)),
           "import_time_risky_calls": calls}
    if "--json" in sys.argv:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    print(f"起点: {start}\n到達する acestep モジュール（eager）: {len(order)}")
    print("\n--- サードパーティ（eager。最初の import 元 / import するモジュール数）")
    for k, v in third.items():
        print(f"  {k:28s} {v[0]}  ({len(set(v))})")
    print("\n--- 関数内でのみ import（lazy）されるサードパーティ")
    print("  " + ", ".join(sorted(set(lazy) - set(third))))
    print("\n--- import 時に実行される、CUDA・モデル・確保の疑いのある呼び出し")
    print("  (なし)" if not calls else "")
    for m, c in calls.items():
        for s in c:
            print(f"  {m}: {s}")


if __name__ == "__main__":
    main()

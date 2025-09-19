#!/usr/bin/env python3
"""
Excel Q&A – Minimal (No‑AI)
---------------------------------
שואל שאלות בסיסיות על קובץ אקסל בלי מודל AI.
תומך ב:
- טעינת קובץ אחד (כל הגיליונות נרשמים כטבלאות DuckDB)
- פקודות מובנות (עברית/אנגלית):
  * schema            – מציג שמות טבלאות ועמודות
  * preview <table>   – תצוגה מקדימה של טבלה
  * top products      – המוצרים הכי נמכרים (לפי revenue/amount)
  * last quarter top  – המוצרים הכי נמכרים ברבעון האחרון
  * revenue by month  – סיכום הכנסות לפי חודש
  * עמודות/סכימה/תצוגה/הכי נמכר/רבעון אחרון/לפי חודש – מילות מפתח בעברית
  * SELECT ...        – אפשר להריץ SQL חופשי (קריאה בלבד)
- CLI אינטראקטיבי: תריץ ותכתוב פקודות. יציאה: quit/exit

התקנות:
    pip install -U duckdb pandas openpyxl rich
הרצה:
    python excel_qa_min.py path/to/file.xlsx
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import pandas as pd
from rich import print as rprint
from rich.table import Table

MAX_ROWS = 1000

class Engine:
    def __init__(self):
        self.conn = duckdb.connect()
        self.registered: Dict[str, pd.DataFrame] = {}

    def load_excel(self, path: Path) -> List[str]:
        if not path.exists():
            raise FileNotFoundError(path)
        xl = pd.ExcelFile(path)
        views = []
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            df = df.copy()
            df.columns = [re.sub(r"[^\w]+", "_", str(c)).strip("_").lower() for c in df.columns]
            name = f"{path.stem}__{sheet}".lower()
            name = re.sub(r"[^a-z0-9_]+", "_", name)
            self.registered[name] = df
            self.conn.register(name, df)
            views.append(name)
        return views

    def schema(self) -> Dict[str, List[str]]:
        return {t: list(df.columns) for t, df in self.registered.items()}

    def preview(self, table: str, n: int = 5) -> pd.DataFrame:
        self._ensure(table)
        return self.conn.execute(f"SELECT * FROM {table} LIMIT {n}").df()

    def ask(self, question: str) -> pd.DataFrame:
        sql = self._question_to_sql(question)
        df = self.conn.execute(sql).df()
        return df.head(MAX_ROWS)

    # --- internals ---
    def _ensure(self, table: str):
        if table not in self.registered:
            raise ValueError(f"Unknown table '{table}'. Known: {list(self.registered)}")

    def _guess_table_and_cols(self):
        # בוחר את הטבלה עם הכי הרבה מאפייני מכירות
        best = None
        best_score = -1
        for t, df in self.registered.items():
            cols = set(df.columns)
            score = 0
            for c in ["date", "order_date", "invoice_date", "תאריך"]:
                if c in cols: score += 2
            for c in ["quantity", "qty", "units", "כמות"]:
                if c in cols: score += 2
            for c in ["amount", "total", "revenue", "price", "sum", "סכום"]:
                if c in cols: score += 2
            for c in ["product", "item", "sku", "מוצר", "שם_מוצר"]:
                if c in cols: score += 1
            if score > best_score:
                best = t
                best_score = score
        if best is None and self.registered:
            best = list(self.registered)[0]
        # ברירות מחדל לשמות עמודות
        df = self.registered[best]
        cols = df.columns
        product = _first_of(cols, ["product","item","sku","מוצר","שם_מוצר"]) or cols[0]
        qty     = _first_of(cols, ["quantity","qty","units","כמות"]) or cols[1 if len(cols)>1 else 0]
        amount  = _first_of(cols, ["amount","total","revenue","price","sum","סכום"]) or qty
        date    = _first_of(cols, ["date","order_date","invoice_date","תאריך"]) or None
        return best, product, qty, amount, date

    def _question_to_sql(self, q: str) -> str:
        ql = q.strip().lower()
        if ql.startswith("select "):
            return _sanitize_select(q)

        table, product, qty, amount, date = self._guess_table_and_cols()

        # בעברית/אנגלית – זיהוי פשוט לפי מילות מפתח
        if any(k in ql for k in ["schema", "columns", "עמודות", "סכימה"]):
            return f"SELECT * FROM {table} LIMIT 5"

        if any(k in ql for k in ["preview", "תצוגה", "הצגה"]):
            return f"SELECT * FROM {table} LIMIT 10"

        if any(k in ql for k in ["last quarter", "רבעון האחרון"]):
            if not date:
                raise ValueError("לא נמצאה עמודת תאריך מתאימה לצורך רבעון אחרון.")
            return f"""
            WITH base AS (
              SELECT *, date_trunc('quarter', {date}) AS qtr FROM {table}
            ), lastq AS (
              SELECT max(qtr) AS q FROM base
            )
            SELECT {product} AS product, SUM({qty}) AS units, SUM({amount}) AS revenue
            FROM base WHERE qtr = (SELECT q FROM lastq)
            GROUP BY 1 ORDER BY revenue DESC, units DESC LIMIT 50
            """

        if any(k in ql for k in ["top products", "הכי נמכר", "המוצרים הכי"]):
            return f"""
            SELECT {product} AS product, SUM({qty}) AS units, SUM({amount}) AS revenue
            FROM {table}
            GROUP BY 1 ORDER BY revenue DESC, units DESC LIMIT 50
            """

        if any(k in ql for k in ["by month", "per month", "לפי חודש", "חודש"]):
            if not date:
                raise ValueError("לא נמצאה עמודת תאריך מתאימה לצורך קיבוץ לפי חודש.")
            return f"""
            SELECT date_trunc('month', {date}) AS month, SUM({amount}) AS revenue, SUM({qty}) AS units
            FROM {table}
            GROUP BY 1 ORDER BY 1
            """

        # ברירת מחדל: תצוגה
        return f"SELECT * FROM {table} LIMIT 10"


def _sanitize_select(sql: str) -> str:
    s = sql.strip().rstrip(";")
    low = s.lower()
    forbidden = ["update ", "insert ", "delete ", "drop ", "alter ", "create ", "attach ", "copy ", "replace "]
    if not low.startswith("select "):
        raise ValueError("רק SELECT מותר.")
    if any(tok in low for tok in forbidden):
        raise ValueError("מותר קריאה בלבד (SELECT).")
    if " limit " not in low:
        s += f" LIMIT {MAX_ROWS}"
    return s


def _first_of(cols: List[str], options: List[str]) -> Optional[str]:
    for o in options:
        if o in cols:
            return o
    return None

# ---------------- CLI ----------------

def _print_df(df: pd.DataFrame, title: str = ""):
    table = Table(title=title or f"Rows: {len(df)}")
    for col in df.columns:
        table.add_column(str(col))
    for _, row in df.iterrows():
        table.add_row(*[str(v) for v in row.values])
    rprint(table)


def repl(engine: Engine):
    rprint("[bold]Loaded.[/bold] Commands: schema | preview <table> | top products | last quarter top | revenue by month | SELECT ... | quit")
    while True:
        try:
            q = input("excel> ").strip()
        except (EOFError, KeyboardInterrupt):
            print() ; break
        if not q:
            continue
        if q.lower() in {"quit", "exit", ":q"}:
            break
        try:
            if q.lower().startswith("preview "):
                _, t = q.split(" ", 1)
                df = engine.preview(t.strip())
                _print_df(df, f"Preview {t.strip()}")
                continue
            if q.lower() in {"schema", "סכימה", "עמודות"}:
                rprint(engine.schema())
                continue
            df = engine.ask(q)
            _print_df(df, "Result") 
        except Exception as e:
            rprint(f"[red]Error:[/red] {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python excel_qa_min.py <excel.xlsx>")
        sys.exit(1)
    path = Path(sys.argv[1])
    eng = Engine()
    views = eng.load_excel(path)
    rprint({"registered": views})
    repl(eng)

if __name__ == "__main__":
    main()

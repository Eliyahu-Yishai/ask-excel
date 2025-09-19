#!/usr/bin/env python3
"""
Excel Q&A – Simple (with optional OpenAI NL→SQL)
------------------------------------------------
- Ask questions over an Excel file using DuckDB.
- Works WITHOUT AI (built-in commands + direct SELECT).
- If OPENAI_API_KEY is set, you can ask in natural language (Heb/Eng).

Install:
    pip install -U duckdb pandas openpyxl rich requests

Run:
    python excel_qa_simple.py path/to/your.xlsx

At the prompt (REPL):
    schema
    preview <table>
    top products
    revenue by month
    last quarter top
    SELECT product, SUM(quantity) ...   (any SELECT)
    מה המוצר הכי נמכר ברבעון האחרון?   (if OPENAI_API_KEY is set)
    quit
"""
import os
import re
import sys
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from constants import OPENAI_API_KEY, OPENAI_MODEL

import duckdb
import pandas as pd
from rich import print as rprint
from rich.table import Table

MAX_ROWS = 1000

def _first_of(cols: List[str], options: List[str]) -> Optional[str]:
    for o in options:
        if o in cols:
            return o
    return None

def _sanitize_select(sql: str) -> str:
    s = sql.strip().rstrip(";")
    low = s.lower()
    if not low.startswith("select "):
        raise ValueError("Only SELECT is allowed.")
    forbidden = [" update ", " insert ", " delete ", " drop ", " alter ", " create ", " attach ", " copy ", " replace "]
    if any(tok in f" {low} " for tok in forbidden):
        raise ValueError("Read-only mode: SELECT statements only.")
    if " limit " not in low:
        s += f" LIMIT {MAX_ROWS}"
    return s

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

    def _ensure(self, table: str):
        if table not in self.registered:
            raise ValueError(f"Unknown table '{table}'. Known: {list(self.registered)}")

    def _guess_table_and_cols(self):
        best, best_score = None, -1
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
                best, best_score = t, score
        if best is None and self.registered:
            best = list(self.registered)[0]
        df = self.registered[best]
        cols = df.columns
        product = _first_of(cols, ["product","item","sku","מוצר","שם_מוצר"]) or cols[0]
        qty     = _first_of(cols, ["quantity","qty","units","כמות"]) or (cols[1] if len(cols)>1 else cols[0])
        amount  = _first_of(cols, ["amount","total","revenue","price","sum","סכום"]) or qty
        date    = _first_of(cols, ["date","order_date","invoice_date","תאריך"])
        return best, product, qty, amount, date

    def _builtins_to_sql(self, q: str) -> Optional[str]:
        ql = q.lower()
        if ql.startswith("select "):
            return _sanitize_select(q)
        table, product, qty, amount, date = self._guess_table_and_cols()
        if any(k in ql for k in ["schema", "columns", "עמודות", "סכימה"]):
            return f"SELECT * FROM {table} LIMIT 5"
        if ql.startswith("preview "):
            return f"SELECT * FROM {ql.split(' ',1)[1].strip()} LIMIT 10"
        if any(k in ql for k in ["top products", "הכי נמכר", "המוצרים הכי"]):
            return f"""
            SELECT {product} AS product, SUM({qty}) AS units, SUM({amount}) AS revenue
            FROM {table}
            GROUP BY 1 ORDER BY revenue DESC, units DESC LIMIT 50
            """
        if any(k in ql for k in ["by month", "per month", "לפי חודש", "חודש"]):
            if not date:
                raise ValueError("No date column for monthly grouping.")
            return f"""
            SELECT date_trunc('month', {date}) AS month, SUM({amount}) AS revenue, SUM({qty}) AS units
            FROM {table}
            GROUP BY 1 ORDER BY 1
            """
        if any(k in ql for k in ["last quarter", "רבעון האחרון"]):
            if not date:
                raise ValueError("No date column for quarter grouping.")
            return f"""
            WITH base AS (
              SELECT *, date_trunc('quarter', {date}) AS qtr FROM {table}
            ), lastq AS (SELECT max(qtr) AS q FROM base)
            SELECT {product} AS product, SUM({qty}) AS units, SUM({amount}) AS revenue
            FROM base WHERE qtr = (SELECT q FROM lastq)
            GROUP BY 1 ORDER BY revenue DESC, units DESC LIMIT 50
            """
        return None

    # --- AI ---
    def nl_to_sql(self, question: str) -> str:
        if question.strip().lower().startswith("select "):
            return _sanitize_select(question)
        api_key = OPENAI_API_KEY
        print(f"Using OpenAI model: {OPENAI_MODEL or 'gpt-4o'}")
        print(OPENAI_API_KEY)
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set (AI mode).")
        schema_json = json.dumps({t: list(df.columns) for t, df in self.registered.items()}, ensure_ascii=False, indent=2)
        system = (
             "You translate natural-language questions (Hebrew or English) into a SINGLE DuckDB-compatible SQL query.\n"
            "Return ONLY raw SQL, with no commentary, no code fences, no prefixes.\n"
            "The SQL must start with SELECT or WITH.\n"
            "Constraints:\n"
            "- READ-ONLY: SELECT only (no DDL/DML)\n"
            "- Use the provided table/column names exactly (snake_case, lowercase)\n"
            "- If no LIMIT present, add LIMIT 1000\n"
            "- Use date_trunc for monthly/quarterly grouping when applicable\n"
        )
        user = f"SCHEMA:\\n{schema_json}\\n\\nQuestion:\\n{question}\\nReturn a single SELECT."
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": OPENAI_MODEL or "gpt-4o",
            "messages": [{"role":"system","content":system},{"role":"user","content":user}],
            "temperature": 0
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=45)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        m = re.search(r"```sql\\s*(.*?)```", content, re.S|re.I)
        sql = m.group(1).strip() if m else content
        return _sanitize_select(sql)

    def run_sql(self, sql: str) -> pd.DataFrame:
        return self.conn.execute(sql).df().head(MAX_ROWS)

def _print_df(df: pd.DataFrame, title: str = ""):
    table = Table(title=title or f"Rows: {len(df)}")
    for col in df.columns:
        table.add_column(str(col))
    for _, row in df.iterrows():
        table.add_row(*[str(v) for v in row.values])
    rprint(table)

def repl(engine: Engine):
    ai_on = bool(OPENAI_API_KEY)
    rprint(f"[bold]Loaded.[/bold] AI = {'ON' if ai_on else 'OFF'} | Commands: schema | preview <table> | top products | revenue by month | last quarter top | SELECT ... | quit")
    while True:
        try:
            q = input("excel> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not q:
            continue
        if q.lower() in {"quit","exit",":q"}:
            break
        try:
            # Try built-ins first
            sql = engine._builtins_to_sql(q)
            if sql is None:
                if ai_on:
                    sql = engine.nl_to_sql(q)
                else:
                    raise ValueError("Unrecognized command. Use built-ins or write a SELECT.")
            df = engine.run_sql(sql)
            _print_df(df, "Result")
        except Exception as e:
            rprint(f"[red]Error:[/red] {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python excel_qa_simple.py <excel.xlsx>"); sys.exit(1)
    path = Path(sys.argv[1])
    eng = Engine()
    views = eng.load_excel(path)
    rprint({"registered": views})
    repl(eng)

if __name__ == "__main__":
    main()

# Ask Excel - Excel Q&A Tool

כלי לשאילת שאלות בעברית ואנגלית על קבצי Excel באמצעות AI, עם ממשק שורת פקודה אינטראקטיבי.

## תכונות

- **שאלות בשפה טבעית**: שאול שאלות בעברית או אנגלית על הנתונים שלך
- **תמיכה ב-Excel**: טעינה וניתוח של קבצי `.xlsx`
- **AI מתקדם**: שימוש ב-OpenAI GPT לתרגום שאלות ל-SQL
- **ממשק אינטראקטיבי**: REPL נוח עם Rich formatting
- **שאילתות SQL ישירות**: אפשרות להריץ SQL ישירות על הנתונים

## דרישות מערכת

- Python 3.8+
- OpenAI API Key (אופציונלי - למצב AI)

## התקנה

1. **שכפל את הפרויקט:**
   ```bash
   git clone https://github.com/Eliyahu-Yishai/ask-excel.git
   cd ask-excel
   ```

2. **התקן תלות:**
   ```bash
   pip install pandas openpyxl duckdb requests rich
   ```

3. **הגדר מפתח API (אופציונלי):**
   
   צור קובץ `constants.py` (או עדכן את הקיים):
   ```python
   OPENAI_API_KEY = "sk-your-openai-api-key-here"
   OPENAI_MODEL = "gpt-4o"  # או "gpt-4o-mini"
   ```

   **חשוב:** אל תשתף את המפתח שלך! הוסף את `constants.py` ל-`.gitignore`

## שימוש

```bash
python excel_qa_simple.py path/to/your/file.xlsx
```

### דוגמאות שאלות

**בעברית:**
- "מה הכנסות השנה?"
- "תן לי את 5 המוצרים הכי רווחיים"
- "כמה הזמנות היו בחודש האחרון?"
- "איזה לקוח קנה הכי הרבה?"

**באנגלית:**
- "What are the top selling products?"
- "Show me revenue by month"
- "Which region has the highest sales?"
- "What's the average order value?"

### פקודות מיוחדות

- `schema` - הצג את מבנה הטבלאות
- `preview <table_name>` - הצג דוגמה מטבלה
- `SELECT ...` - הרץ SQL ישירות
- `quit` - יציאה

## מבנה הפרויקט

```
ask-excel/
├── excel_qa_simple.py    # הקובץ הראשי
├── excel_qa.py          # גרסה מתקדמת יותר
├── constants.py         # הגדרות API (לא בגיט)
├── sales_demo.xlsx      # קובץ דוגמה
└── README.md           # המדריך הזה
```

## איך זה עובד?

1. **טעינת Excel**: הכלי טוען את קובץ ה-Excel ומפרק אותו לטבלאות
2. **עיבוד שאלות**: כשאתה שואל שאלה, הכלי משתמש ב-AI לתרגם אותה ל-SQL
3. **הרצת שאילתה**: השאילתה רצה על הנתונים באמצעות DuckDB
4. **הצגת תוצאות**: התוצאות מוצגות בפורמט נוח וקריא

## פתרון בעיות

### "OPENAI_API_KEY not set"
- ודא שהקובץ `constants.py` קיים ומכיל את המפתח שלך
- אלטרנטיבה: הגדר משתנה סביבה: `set OPENAI_API_KEY=your-key`

### "File not found"
- ודא שנתיב הקובץ נכון
- הקובץ חייב להיות בפורמט `.xlsx`

### שגיאות SQL
- נסה לנסח את השאלה אחרת
- השתמש בפקודה `schema` לראות את המבנה הזמין
- הרץ SQL ישירות לבדיקה

## תמיכה

- 🐛 **באגים**: פתח issue ב-GitHub
- 💡 **רעיונות**: הצע features חדשים
- 📧 **שאלות**: צור קשר דרך GitHub

## רישיון

MIT License - ראה קובץ LICENSE לפרטים

---

**עשה בישראל 🇮🇱 | Made in Israel**
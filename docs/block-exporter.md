# Block Exporter — מדריך שימוש מהיר

מסמך זה מסביר איך נוצרים “בלוקים” מהדו"ח (Account Blocks), איפה נשמרים הקבצים, מה הפורמט שלהם, ואילו לוגים מאפשרים ניטור מהיר.

---

## סדר ריצה — היכן זה נכנס בפייפליין

1. **אורקסטרייטור** (`backend/core/orchestrators.py`)
   - מיד לאחר בדיקות בטיחות PDF ולפני כל ניתוח נוסף:
     ```python
     export_account_blocks(session_id, pdf_path)
     ```
   - אם אין בלוקים: המודול מרים `ValueError("No blocks extracted")` והפייפליין נעצר (Fail-Fast).

2. **אנלייזר** (`backend/core/logic/report_analysis/analyze_report.py`)
   - לא כותב בלוקים מחדש.
   - טוען בלוקים שמורים לצורך תאימות לאחור:
     ```python
     fbk_blocks = load_account_blocks(session_id)
     result["fbk_blocks"] = fbk_blocks
     result["blocks_by_account_fuzzy"] = build_block_fuzzy(fbk_blocks) if fbk_blocks else {}
     ```

---

## היכן הקוד יושב

- יצוא/טעינה/העשרה:
  - `backend/core/logic/report_analysis/block_exporter.py`
    - `export_account_blocks(session_id, pdf_path) -> list[dict]`
    - `load_account_blocks(session_id) -> list[dict]`
    - `enrich_block(blk: dict) -> dict` — מוסיף `fields` מובנים לכל לשכה.

- עזרי ניתוח/טקסט:
  - `backend/core/logic/utils/text_parsing.py` — `extract_account_blocks`
  - `backend/core/logic/report_analysis/report_parsing.py` — `detect_bureau_order`, `build_block_fuzzy`

---

## מיקומי קבצים בדיסק

לאחר ריצה מוצלחת תראו:


traces/
blocks/
<SESSION_ID>/
_index.json
block_01.json
block_02.json
...


- `_index.json` — **רק**: `[{ "i": <int>, "heading": <str>, "file": "<path>" }, ...]`
- `block_XX.json` — בלוקים מועשרים (ראו פורמט בהמשך).

> שימו לב: פורמט `_index.json` נשמר מינימלי לשמירת תאימות; אין להוסיף בו מפתחות חדשים.

---

## פורמט הבלוק (enriched)

כל קובץ `block_XX.json` מכיל:
```json
{
  "heading": "AMEX",
  "lines": [
    "AMEX",
    "Transunion® Experian® Equifax®",
    "Account # 3499********** 3499********** -3499***********",
    "High Balance: $10,964 $10,964 $0",
    "Date Opened: 14.9.2018 1.9.2018 1.9.2018",
    "Payment Status: Current Current Current",
    "Creditor Remarks: -- -- Credit card"
  ],
  "fields": {
    "transunion": {
      "account_number_display": "3499**********",
      "high_balance": "$10,964",
      "date_opened": "14.9.2018",
      "payment_status": "Current",
      "creditor_remarks": ""
      /* ... שדות מוכרים נוספים, ערך ריק אם לא נמצא ... */
    },
    "experian": {
      "account_number_display": "3499**********",
      "high_balance": "$10,964",
      "date_opened": "1.9.2018",
      "payment_status": "Current",
      "creditor_remarks": ""
    },
    "equifax": {
      "account_number_display": "3499***********",
      "high_balance": "$0",
      "date_opened": "1.9.2018",
      "payment_status": "Current",
      "creditor_remarks": "Credit card"
    }
  }
}

מיפוי מפתחות לשדות

השורות ב־lines מפוענחות לפי תוויות סטנדרטיות (לא רגיש לאותיות רישיות), לדוגמה:

Account # → account_number_display

High Balance → high_balance

Date Opened → date_opened

Payment Status → payment_status

Creditor Remarks → creditor_remarks

(נוסף: last_verified, date_of_last_activity, date_reported, balance_owed, closed_date, account_rating, account_description, dispute_status, creditor_type, account_status, payment_amount, last_payment, term_length, past_due_amount, account_type, payment_frequency, credit_limit)

ערך שאין לו נתון או שמודפס כ־-- נשמר כמחרוזת ריקה.

לוגים ואיך לקרוא אותם

ביצוא:

תחילת התהליך:
ANZ: export kickoff sid=<SID> file=<PATH>

לפני כתיבה (מטריקות):
ANZ: pre-save fbk=<N_BLOCKS> fuzzy=<N_KEYS> sid=<SID>

סיכום לכל בלוק:
BLOCK: enrichment_summary heading='<HEADING>' tu_filled=<N> ex_filled=<N> eq_filled=<N>

אחרי כתיבה:
ANZ: export blocks sid=<SID> dir=<DIR> files=<COUNT>

טעינה לאחר יצוא (באורקסטרייטור):
ANZ: blocks ready sid=<SID> count=<COUNT>

בכשל:

אין בלוקים (נעצר):
BLOCKS_FAIL_FAST: 0 blocks extracted sid=<SID> file=<PATH>

בדיקת עשן (ידנית)

הרם Redis ו־Celery:

python -m celery -A backend.api.tasks worker --loglevel=info --pool=solo -Q celery,merge,validation,note_style,frontend


העלה דו"ח והפעל תהליך (דרך ה־API/UI/קריאה ישירה לאורקסטרייטור).

אמת שהתקיות והקבצים נוצרו:

traces/blocks/<SESSION>/_index.json
traces/blocks/<SESSION>/block_01.json


בדוק לוגים לפי הסעיף לעיל.

תאימות לאחור

analyze_report עוד מקבל/ממלא result["fbk_blocks"] ו־result["blocks_by_account_fuzzy"], אבל לא מייצא בלוקים מחדש; הוא טוען את מה שכבר נכתב בדיסק.

_index.json נשמר בפורמט הישן (i, heading, file) כדי לא לשבור צרכנים קיימים.

#### תוצרים
- קובץ `docs/block-exporter.md` חדש עם התוכן הנ"ל.

#### DoD
- מפתח חדש מסוגל:
  - להבין איפה נוצרים הבלוקים ומתי הריצה נעצרת על אפס בלוקים.
  - למצוא את הקבצים בדיסק ולקרוא את הפורמט.
  - לזהות בקלות בלוגים שהכול רץ/נכשל.

---

תגיד לי כשאתה רוצה שנעבור למשימה 13.

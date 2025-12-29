# Case-First Debug Runbook

מטרת המסמך: להריץ מקצה-לקצה (PDF → Extractor → Case Builder → Stage-A → Collectors → UI) ולדבג במהירות כשאין Cases.

---

## 1) דרישות מקדימות

- **Redis** רץ לוקאלית (broker + backend):
  - ברירת המחדל שלנו: `redis://localhost:6379/0` (broker) ו‎`redis://localhost:6379/1` (result backend).
- Python venv עם התלויות של הפרויקט מותקנות.
- Node.js לצורך הפרונט (Vite).

> אם אין Redis:
> - **Docker (מומלץ):**
>   ```bash
>   docker run -p 6379:6379 --name dev-redis -d redis:7
>   ```
> - **Windows Service / Redis מקומי:** ודא ש‎redis-server רץ ופותח 6379.

---

## 2) משתני סביבה (Dev)

הגדר את הדגלים האלו **גם** ב‎Celery וגם ב‎Flask (אותן ערכים בשני התהליכים):

```powershell
# Core / AI (אם אין מפתח, אפשר לכבות adjudicator)
$env:OPENAI_API_KEY  = "<your-key>"         # או להשאיר ריק ולכבות adjudicator
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"

# Parsing
$env:RULEBOOK_FALLBACK_ENABLED = "1"

# Case-First strict mode
$env:ONE_CASE_PER_ACCOUNT_ENABLED = "1"
$env:SAFE_MERGE_ENABLED           = "1"
$env:CASE_FIRST_BUILD_REQUIRED    = "1"
$env:DISABLE_PARSER_UI_SUMMARY    = "1"
$env:NORMALIZED_OVERLAY_ENABLED   = "0"

# Broker / Results (Redis)
$env:CELERY_BROKER_URL     = "redis://localhost:6379/0"
$env:CELERY_RESULT_BACKEND = "redis://localhost:6379/1"

# Debug / Telemetry
$env:CASEBUILDER_DEBUG = "1"
$env:METRICS_ENABLED    = "1"

# Legacy/materializer
$env:MATERIALIZER_ENABLE = "0"
```

TTL;DR: אם Celery נופל עם Connection refused ל‎AMQP, סימן שאין Broker. ודאו ש‎CELERY_BROKER_URL מצביע ל‎Redis רץ.

## 3) הרצת התהליכים

### 3.1 Celery Worker
```powershell
$PY = "C:\\venvs\\credit\\Scripts\\python.exe"
cd C:\\dev\\credit-analyzer
# הגדר ENV כמו למעלה בחלון הזה
& $PY -m celery -A backend.api.tasks worker --loglevel=debug --pool=solo -Q celery,merge,validation,note_style,frontend
```

חפשו בלוגים:

- `CASEBUILDER: starting / pre-count / post-count`
- `CASEBUILDER: summary session=<sid> input=<X> upserted=<Y> dropped={...}`
- `METRIC increment/gauge ...`
- `stageA.run.count bureau=EX/EQ/TU`

### 3.2 Flask Backend (פורט 5000)
```powershell
$PY = "C:\\venvs\\credit\\Scripts\\python.exe"
cd C:\\dev\\credit-analyzer
# הגדר ENV כמו למעלה בחלון הזה
$env:FLASK_APP   = "backend.api.app:create_app"
$env:FLASK_DEBUG = "1"
& $PY -m flask --app $env:FLASK_APP run --port 5000 --reload
```

### 3.3 Frontend (Vite)
```powershell
cd C:\\dev\\credit-analyzer\\frontend
# פעם ראשונה:
# npm install
$env:VITE_API_URL = "http://localhost:5000"
npm run dev
```

## 4) הרצה מקצה-לקצה

העלו PDF דרך ה-UI או דרך API/סקрипט העלאה (מייצר session_id חדש).

ודאו ש‎Celery מדפיס `CASEBUILDER: summary ... upserted > 0` ו‎`post-count > 0`.

## 5) בדיקות API מהירות

נניח שה-SID שהתקבל הוא `<sid>`:

```powershell
Invoke-RestMethod "http://localhost:5000/api/cases/<sid>" | ConvertTo-Json -Depth 10
Invoke-RestMethod "http://localhost:5000/api/session/<sid>/logical_index" | ConvertTo-Json -Depth 10
Invoke-RestMethod "http://localhost:5000/api/accounts/<sid>?include_case_ids=1" | ConvertTo-Json -Depth 10
```

בחרו `case_id` אמיתי (לא slug) מהתשובות והביאו “תיק מלא”:

```powershell
Invoke-RestMethod "http://localhost:5000/api/account/<sid>/<case_id>" | ConvertTo-Json -Depth 20
```

מצופה לראות ב‎`/api/account/<sid>/<case_id>`:

- `fields.by_bureau.EX|EQ|TU`
- `artifacts.stageA_detection.EX|EQ|TU`
- (אינדיקציה לדגלים ב-meta/flags אם נחשפים)

## 6) טרבלשוט מהיר

### Broker / Celery

**שגיאה:** `[Errno 111] Connection refused` / AMQP ברירת מחדל
**פתרון:** הרם Redis מקומי והגדר `CELERY_BROKER_URL` ו‎`CELERY_RESULT_BACKEND` ל‎Redis. ודא שה-port 6379 פתוח.

**ה-Worker לא רואה ENV:** הגדר את כל ה-ENV באותו חלון PowerShell שבו אתה מריץ Celery.

### Case Builder

`post-count == 0`:

בדקו לוג `CASEBUILDER: summary ... dropped={...}`:

- `missing_logical_key` → בדקו issuer נורמל (Task 3) ו/או opened_date.
- `min_fields` → תנאי הכיסוי מחמיר מדי (שקלו להקל או להדפיס אילו שדות חסרים).
- `write_error` → יש stacktrace; טפלו בשגיאת upsert.
- אין summary → ודאו `CASEBUILDER_DEBUG=1` ושקריאה ל‎`build_account_cases(session_id)` מתבצעת אחרי הניתוח.

### Stage-A

**שגיאה:** `no_account_cases`
**סיבה:** אין Cases; חייבים לתקן את הבילדר/האקסטרקטור.
**בדיקה:** `/api/cases/<sid>` ו‎`/api/session/<sid>/logical_index`.

### טסטים / סביבת Dev

- `PyMuPDF segfault`: הימנעו מטסטים התלויים ב-PDF; הריצו רק unit מבודדים.
- `OPENAI_API_KEY` חסר: כבו adjudicator או הזרימו key בסביבתכם.

## 7) “Before / After” — מה נחשב הצלחה

Celery מציג:

- `CASEBUILDER: starting, pre-count, summary (input>0, upserted>0), post-count>0.`
- אירועי `METRIC` עבור `casebuilder/casestore/stageA`.

API:

- `/api/cases/<sid>` מחזיר לפחות Case אחד.
- `/api/accounts/<sid>?include_case_ids=1` מחזיר account_id אמיתיים (לא slug).
- `/api/account/<sid>/<case_id>` מציג `fields.by_bureau.*` וגם `artifacts.stageA_detection.*`.

הפרונט מציג את הרשימה מאותו מקור (collectors), ללא `parser_aggregated`.

## 8) נספח: פקודות Curl (אלטרנטיבה ל-PowerShell)
```bash
SID="<sid>"
curl -s "http://localhost:5000/api/cases/$SID" | jq
curl -s "http://localhost:5000/api/session/$SID/logical_index" | jq
curl -s "http://localhost:5000/api/accounts/$SID?include_case_ids=1" | jq
CASE_ID="<pick-from-above>"
curl -s "http://localhost:5000/api/account/$SID/$CASE_ID" | jq
```

---

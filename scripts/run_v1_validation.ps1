# שמירת תוצאות בתיקיית .artifacts
New-Item -ItemType Directory -Force -Path .artifacts | Out-Null

# לעקוף יצירת PDF כדי שלא ייפול על חוסר ב-wkhtmltopdf
$env:DISABLE_PDF_RENDER = "true"

# 1) רשימת כל הבדיקות (בלי להריץ)
pytest --collect-only -q --capture=no | Out-File .artifacts\test_inventory.txt

# 2) ריצה של כל הבדיקות ושמירת תוצאה לקובץ
#    שים לב ל- --capture=no  ← זה עוקף את הבאג של "Bad file descriptor"
pytest -q --disable-warnings -ra --capture=no 2>&1 | Out-File .artifacts\test_run_all_skip_pdf.txt

# 3) הצגת הסיכום על המסך (30 שורות אחרונות)
Get-Content .artifacts\test_run_all_skip_pdf.txt -Tail 30

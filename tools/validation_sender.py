import os, sys, json, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, argparse

def getenv(name, default=None, cast=str):
    v = os.getenv(name, default)
    if v is None: return None
    if cast is int:
        try: return int(str(v).strip())
        except: return int(default) if default is not None else None
    if cast is float:
        try: return float(str(v).strip())
        except: return float(default) if default is not None else None
    if cast is list:
        s = str(v).strip()
        if not s: return []
        return [x.strip() for x in s.split(',')]
    return str(v)

OPENAI_BASE_URL   = getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
OPENAI_API_KEY    = getenv('OPENAI_API_KEY', '')
OPENAI_PROJECT_ID = getenv('OPENAI_PROJECT_ID', None)
MODEL             = getenv('VALIDATION_MODEL') or getenv('AI_MODEL') or 'gpt-4o-mini'
RESP_FMT          = getenv('AI_RESPONSE_FORMAT', 'json_object')
TEMPERATURE       = float(getenv('AI_TEMPERATURE', '0') or 0)
TOP_P             = float(getenv('AI_TOP_P', '1') or 1)
MAX_TOKENS        = int(getenv('AI_MAX_TOKENS', '400') or 400)

HTTP_CONNECT_TO   = int(getenv('AI_HTTP_CONNECT_TIMEOUT', '10') or 10)
HTTP_READ_TO      = int(getenv('AI_HTTP_READ_TIMEOUT', '40') or 40)

MAX_WORKERS       = int(getenv('SENDER_CONCURRENCY', '2') or 2)
RPS               = float(getenv('RATE_LIMIT_RPS', '2') or 2)
MAX_INFLIGHT      = int(getenv('RATE_LIMIT_MAX_INFLIGHT', '4') or 4)
RETRY_MAX         = int(getenv('SENDER_RETRY_MAX', '3') or 3)
RETRY_BACKOFFS_MS = getenv('SENDER_RETRY_BACKOFF_MS', '100,250,500', cast=list)
RETRY_BACKOFFS    = [int(x)/1000.0 for x in RETRY_BACKOFFS_MS] or [0.1, 0.25, 0.5]

WRITE_JSONL       = getenv('VALIDATION_WRITE_JSONL', '1') in ('1','true','True')
WRITE_JSON        = getenv('VALIDATION_WRITE_JSON', '1') in ('1','true','True')

session = requests.Session()
session.headers.update({'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json'})
if OPENAI_PROJECT_ID:
    session.headers['OpenAI-Project'] = OPENAI_PROJECT_ID

rate_lock = threading.Lock()
last_request_ts = [0.0]
inflight_sem = threading.Semaphore(MAX_INFLIGHT)

def rate_limit():
    if RPS <= 0: return
    min_gap = 1.0 / RPS
    with rate_lock:
        now = time.time()
        wait = last_request_ts[0] + min_gap - now
        if wait > 0: time.sleep(wait)
        last_request_ts[0] = time.time()

def call_openai(system_txt, user_obj):
    rf = {'type':'json_object'} if RESP_FMT == 'json_object' else None
    body = {
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': system_txt or ''},
            {'role': 'user',   'content': json.dumps(user_obj, ensure_ascii=False)}
        ],
        'temperature': TEMPERATURE,
        'top_p': TOP_P,
        'max_tokens': MAX_TOKENS,
    }
    if rf: body['response_format'] = rf
    url = OPENAI_BASE_URL.rstrip('/') + '/chat/completions'
    for attempt in range(RETRY_MAX + 1):
        try:
            rate_limit()
            with inflight_sem:
                resp = session.post(url, json=body, timeout=(HTTP_CONNECT_TO, HTTP_READ_TO))
            if resp.status_code == 200:
                data = resp.json()
                content = data['choices'][0]['message'].get('content','')
                return {'ok': True, 'content': content, 'raw': data}
            else:
                err = f'HTTP {resp.status_code} {resp.text[:500]}'
        except Exception as e:
            err = f'EXC {repr(e)}'
        if attempt < RETRY_MAX:
            time.sleep(RETRY_BACKOFFS[min(attempt, len(RETRY_BACKOFFS)-1)])
        else:
            return {'ok': False, 'error': err}

def process_pack_line(line_obj):
    prompt = line_obj.get('prompt', {})
    system_txt = prompt.get('system', '') or ''
    user_obj   = prompt.get('user', {}) or {}
    res = call_openai(system_txt, user_obj)
    out = {'input_id': line_obj.get('id'), 'ok': res.get('ok')}
    if res.get('ok'):
        out['model_content'] = res.get('content')
        try: out['model_json'] = json.loads(res['content'])
        except: pass
    else:
        out['error'] = res.get('error')
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sid', required=True)
    ap.add_argument('--base', default='.')
    ap.add_argument('--dry-run', type=int, default=0)
    args = ap.parse_args()

    base = Path(args.base).resolve()
    idx_path = base / 'runs' / args.sid / 'ai_packs' / 'validation' / 'index.json'
    if not idx_path.exists():
        print('INDEX_MISSING', idx_path); sys.exit(2)

    index = json.loads(idx_path.read_text(encoding='utf-8'))
    packs = index.get('packs', [])
    print(f'INDEX_OK packs={len(packs)}')

    for p in packs:
        pack_path   = Path(p['pack_path'])
        result_jsonl_path = Path(p['result_jsonl_path'])
        result_json_path  = Path(p['result_path'])
        result_summary_path = Path(p.get('result_summary_path', str(result_json_path)))

        if args.dry_run:
            print(f"DRY: would process account={p.get('account_id')} lines={p.get('lines')} pack={pack_path}")
            continue

        lines = []
        with pack_path.open('r', encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if not ln: continue
                try: lines.append(json.loads(ln))
                except: pass

        result_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        result_json_path.parent.mkdir(parents=True, exist_ok=True)
        result_summary_path.parent.mkdir(parents=True, exist_ok=True)

        outs = []
        errors = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(process_pack_line, lo) for lo in lines]
            for fu in as_completed(futs):
                o = fu.result(); outs.append(o)
                if not o.get('ok'): errors += 1

        if WRITE_JSONL:
            with result_jsonl_path.open('w', encoding='utf-8') as w:
                for o in outs: w.write(json.dumps(o, ensure_ascii=False) + '\n')

        summary_obj = {
            'sid': args.sid,
            'account_id': p.get('account_id'),
            'lines_in': len(lines),
            'lines_ok': sum(1 for o in outs if o.get('ok')),
            'lines_err': errors,
            'result_jsonl': str(result_jsonl_path),
        }
        if WRITE_JSON:
            with result_json_path.open('w', encoding='utf-8') as w:
                json.dump(summary_obj, w, ensure_ascii=False, indent=2)
            if str(result_summary_path) != str(result_json_path):
                with result_summary_path.open('w', encoding='utf-8') as w:
                    json.dump(summary_obj, w, ensure_ascii=False, indent=2)

        print(f"PACK_DONE account={p.get('account_id')} ok={summary_obj['lines_ok']}/{summary_obj['lines_in']} err={summary_obj['lines_err']} -> {result_jsonl_path.name}")

if __name__ == '__main__':
    if not OPENAI_API_KEY:
        print('ERROR: OPENAI_API_KEY missing', file=sys.stderr); sys.exit(9)
    main()

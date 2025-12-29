import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

function extractSid(argv, env) {
  let sid = env.npm_config_sid;
  const remaining = [];

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--sid') {
      const next = argv[i + 1];
      if (next && !next.startsWith('--')) {
        sid = next;
        i += 1;
        continue;
      }
    } else if (arg?.startsWith('--sid=')) {
      sid = arg.slice('--sid='.length);
      continue;
    }

    remaining.push(arg);
  }

  return { sid, remaining };
}

const argv = process.argv.slice(2);
const { sid, remaining } = extractSid(argv, process.env);

if (!sid) {
  console.error('Usage: npm run review:dev -- --sid=<run-sid>');
  process.exitCode = 1;
  process.exit();
}

const encodedSid = encodeURIComponent(sid);
const openPath = `/runs/${encodedSid}/review`;

const viteBinUrl = new URL('../node_modules/vite/bin/vite.js', import.meta.url);
const viteBinPath = fileURLToPath(viteBinUrl);

const child = spawn(
  process.execPath,
  [viteBinPath, '--open', openPath, ...remaining],
  {
    stdio: 'inherit',
    env: {
      ...process.env,
      VITE_REVIEW_DEV_SID: sid,
    },
  },
);

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});

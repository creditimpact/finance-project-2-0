const rawBase =
  (process.env.VITE_API_BASE_URL || process.env.VITE_API_URL || '').trim()

const fallbackBase = 'http://127.0.0.1:5000'
const base = (rawBase ? rawBase : fallbackBase).replace(/\/+$/, '')

if (!rawBase) {
  console.warn('[smoke] VITE_API_BASE_URL not set, falling back to', fallbackBase)
}

const apiUrl = (path) => `${base}${path.startsWith('/') ? path : `/${path}`}`

const endpoints = ['/api/health', '/api/runs/last']

const decodeBody = async (response) => {
  const contentType = response.headers.get('content-type') || ''
  try {
    if (contentType.includes('application/json')) {
      return await response.json()
    }
    const text = await response.text()
    return text.length <= 512 ? text : `${text.slice(0, 512)}…`
  } catch (err) {
    return `Failed to read body: ${err instanceof Error ? err.message : String(err)}`
  }
}

const probe = async (path) => {
  const url = apiUrl(path)
  const label = `[smoke] GET ${url}`
  try {
    const response = await fetch(url, {
      headers: {
        Accept: 'application/json, text/plain;q=0.8, */*;q=0.2',
      },
    })
    const body = await decodeBody(response)
    const statusLine = `${response.status}${response.ok ? '' : ' (error)'}`
    console.log(`${label} → ${statusLine}`)
    if (typeof body === 'string') {
      console.log(body)
    } else {
      console.dir(body, { depth: 5 })
    }
  } catch (err) {
    console.error(`${label} failed:`, err instanceof Error ? err.message : err)
  }
}

for (const endpoint of endpoints) {
  await probe(endpoint)
}

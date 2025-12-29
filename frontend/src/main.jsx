import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './devtools/setupFrontendReviewMock'
import './index.css'
import App from './App.jsx'
import AppErrorBoundary from './components/AppErrorBoundary'
import { ToastProvider } from './components/ToastProvider'
import { API_BASE_CONFIGURED, API_BASE_URL } from './api.ts'

const proxyActive = Boolean(import.meta.env?.DEV)
const apiBaseSummary = API_BASE_CONFIGURED ? API_BASE_URL : '(not configured)'
const proxySummary = proxyActive ? 'active (Vite dev server proxy)' : 'inactive'

console.info(
  `[finance-platform] API base: ${apiBaseSummary}; Proxy: ${proxySummary}`
)

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <AppErrorBoundary>
      <ToastProvider>
        <App />
      </ToastProvider>
    </AppErrorBoundary>
  </StrictMode>,
)

import { Component, type ErrorInfo, type ReactNode } from 'react'

interface AppErrorBoundaryProps {
  children: ReactNode
}

interface AppErrorBoundaryState {
  hasError: boolean
}

class AppErrorBoundary extends Component<AppErrorBoundaryProps, AppErrorBoundaryState> {
  state: AppErrorBoundaryState = { hasError: false }

  static getDerivedStateFromError(): AppErrorBoundaryState {
    return { hasError: true }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error in AppErrorBoundary:', error, errorInfo)
  }

  private handleReload = () => {
    window.location.reload()
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          role="alert"
          style={{
            backgroundColor: '#fee2e2',
            border: '1px solid #ef4444',
            borderRadius: '0.5rem',
            color: '#991b1b',
            fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
            margin: '1.5rem auto',
            maxWidth: '40rem',
            padding: '1.5rem',
            textAlign: 'left',
          }}
        >
          <h1 style={{ fontSize: '1.5rem', fontWeight: 600, marginBottom: '0.75rem' }}>Something went wrong</h1>
          <p style={{ margin: 0, lineHeight: 1.5 }}>
            The app encountered an unexpected error. Please try reloading the page. If the problem
            persists, contact support.
          </p>
          <button
            type="button"
            onClick={this.handleReload}
            style={{
              backgroundColor: '#991b1b',
              border: 'none',
              borderRadius: '0.375rem',
              color: '#fff',
              cursor: 'pointer',
              fontWeight: 600,
              marginTop: '1.25rem',
              padding: '0.5rem 1rem',
            }}
          >
            Reload app
          </button>
        </div>
      )
    }

    return this.props.children
  }
}

export default AppErrorBoundary

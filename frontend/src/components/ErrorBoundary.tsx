import { Component, ErrorInfo, ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  error?: Error;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: undefined };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error in ErrorBoundary:', error, errorInfo);
  }

  private resetError = () => {
    this.setState({ error: undefined });
  };

  private reloadPage = () => {
    window.location.reload();
  };

  render() {
    const { error } = this.state;

    if (error) {
      return (
        <div
          role="alert"
          style={{
            backgroundColor: '#fee2e2',
            border: '1px solid #ef4444',
            borderRadius: '0.5rem',
            color: '#991b1b',
            padding: '1.5rem',
            margin: '1.5rem',
            fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
          }}
        >
          <h1 style={{ fontSize: '1.5rem', fontWeight: 600, marginBottom: '0.75rem' }}>Something went wrong</h1>
          <p style={{ margin: 0, lineHeight: 1.5 }}>
            The page hit an unexpected error. You can try again or reload the app. If the problem
            continues, please contact support with the details below.
          </p>
          <pre
            style={{
              backgroundColor: '#fff',
              borderRadius: '0.375rem',
              marginTop: '1rem',
              padding: '0.75rem',
              overflowX: 'auto',
              fontSize: '0.875rem',
            }}
          >
            {error.message}
          </pre>
          <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1.25rem' }}>
            <button
              type="button"
              onClick={this.resetError}
              style={{
                backgroundColor: '#991b1b',
                border: 'none',
                borderRadius: '0.375rem',
                color: '#fff',
                cursor: 'pointer',
                fontWeight: 600,
                padding: '0.5rem 1rem',
              }}
            >
              Try again
            </button>
            <button
              type="button"
              onClick={this.reloadPage}
              style={{
                backgroundColor: '#fff',
                border: '1px solid #991b1b',
                borderRadius: '0.375rem',
                color: '#991b1b',
                cursor: 'pointer',
                fontWeight: 600,
                padding: '0.5rem 1rem',
              }}
            >
              Reload app
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;

import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import HomePage from './pages/HomePage';
import UploadPage from './pages/UploadPage';
import StatusPage from './pages/StatusPage';
import ReviewPage from './pages/ReviewPage';
import RunReviewPage from './pages/RunReviewPage';
import RunReviewCompletePage from './pages/RunReviewCompletePage';
import AccountsPage from './pages/Accounts';
import RunDebugPage from './pages/RunDebugPage';
import { API_BASE_CONFIGURED } from './api.ts';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      {!API_BASE_CONFIGURED && (
        <div className="api-base-warning" role="alert">
          API base URL is not configured. Create frontend/.env.local with VITE_API_BASE_URL=…
        </div>
      )}
      <nav>
        <Link to="/">Home</Link>
        <Link to="/upload">Upload</Link>
      </nav>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/status" element={<StatusPage />} />
        <Route path="/review" element={<ReviewPage />} />
        <Route path="/runs/:sid/review" element={<RunReviewPage />} />
        <Route path="/runs/:sid/review/complete" element={<RunReviewCompletePage />} />
        <Route path="/runs/:sid/accounts" element={<AccountsPage />} />
        <Route path="/debug/run/:sid" element={<RunDebugPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;

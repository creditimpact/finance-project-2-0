import { Link } from 'react-router-dom';

export default function HomePage() {
  return (
    <div className="container">
      <h1>Credit Report Processing</h1>
      <p>Upload your credit report PDF to begin the analysis process.</p>
      <Link to="/upload">Get Started</Link>
    </div>
  );
}

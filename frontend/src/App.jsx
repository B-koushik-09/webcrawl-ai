import React, { useState, useEffect } from 'react';
import ChatInterface from './components/ChatInterface';
import AdminPanel from './components/AdminPanel';
import { checkHealth, getIndexStatus, getIndexStats } from './services/api';
import './App.css';

/**
 * VNRVJIET AI - Main Application
 * Intelligent College Website Knowledge Assistant
 */
function App() {
  const [showAdmin, setShowAdmin] = useState(false);
  const [isBackendOnline, setIsBackendOnline] = useState(null);
  const [backendInfo, setBackendInfo] = useState(null);
  const [indexingStatus, setIndexingStatus] = useState(null);
  const [indexStats, setIndexStats] = useState(null);

  // Check backend health on mount
  // Initial data load and health monitoring
  useEffect(() => {
    checkBackendHealth();
    loadIndexData();
    const healthInterval = setInterval(checkBackendHealth, 30000); // Health check every 30s
    return () => clearInterval(healthInterval);
  }, []);

  // Poll index status ONLY when indexing is active
  useEffect(() => {
    if (!indexingStatus) return; // Wait for initial load

    const isActiveIndexing = ['scraping', 'processing_pdfs', 'indexing', 'downloading_pdfs'].includes(indexingStatus.stage);

    if (isActiveIndexing) {
      // Poll every 2 seconds during active indexing
      const statusInterval = setInterval(loadIndexData, 2000);
      return () => clearInterval(statusInterval);
    }
    // No polling when idle - badge updates on page refresh or manual actions
  }, [indexingStatus?.stage]);

  const checkBackendHealth = async () => {
    try {
      const health = await checkHealth();
      setIsBackendOnline(true);
      setBackendInfo(health);
    } catch (err) {
      setIsBackendOnline(false);
      setBackendInfo(null);
    }
  };

  const loadIndexData = async () => {
    try {
      const [status, stats] = await Promise.all([
        getIndexStatus(),
        getIndexStats()
      ]);
      setIndexingStatus(status);
      setIndexStats(stats);
    } catch (err) {
      console.error('Failed to load index data:', err);
    }
  };

  // Determine if currently indexing
  const isIndexing = indexingStatus &&
    ['scraping', 'processing_pdfs', 'indexing', 'downloading_pdfs'].includes(indexingStatus.stage);

  // Calculate progress percentage
  const getProgressPercent = () => {
    if (!indexingStatus || !indexingStatus.total || indexingStatus.total === 0) return 0;
    return Math.min(Math.round((indexingStatus.current / indexingStatus.total) * 100), 100);
  };

  // Determine index badge status
  const getIndexBadgeStatus = () => {
    if (isIndexing) return 'indexing';
    if (indexStats?.has_index) return 'ready';
    return 'not-ready';
  };

  return (
    <div className="app">
      {/* Navigation Bar */}
      <nav className="navbar">
        <div className="nav-container">
          <div className="nav-brand">
            <div className="brand-logo-container">
              <img src="/vnrvjiet-logo.png" alt="VNRVJIET Logo" className="brand-logo" />
            </div>
            <span className="brand-text">VNRVJIET AI</span>
          </div>

          <div className="nav-actions">
            {/* Backend Status Indicator */}
            <div className={`status-indicator ${isBackendOnline ? 'online' : isBackendOnline === false ? 'offline' : 'checking'}`}>
              <span className="status-dot"></span>
              <span className="status-text">
                {isBackendOnline ? 'Connected' : isBackendOnline === false ? 'Offline' : 'Checking...'}
              </span>
            </div>

            {/* Index Status Badge - Dynamic based on indexing state */}
            <div className={`index-badge ${getIndexBadgeStatus()}`}>
              {isIndexing ? (
                <>
                  <span className="indexing-spinner">⏳</span>
                  <span className="indexing-text">Indexing {getProgressPercent()}%</span>
                  <div className="mini-progress-bar">
                    <div className="mini-progress-fill" style={{ width: `${getProgressPercent()}%` }}></div>
                  </div>
                </>
              ) : indexStats?.has_index ? (
                <>
                  <span className="indexed-icon">✅</span>
                  <span className="indexed-text">Indexed</span>
                  <span className="indexed-count">{indexStats?.pages_scraped || 0} pages</span>
                </>
              ) : (
                <>
                  <span>⚠️</span>
                  <span>Not Indexed</span>
                </>
              )}
            </div>

            {/* Admin Button */}
            <button
              className="admin-btn"
              onClick={() => setShowAdmin(true)}
            >
              <span>⚙️</span>
              <span className="admin-btn-text">Admin</span>
            </button>

            {/* Clear Chat Button */}
            <button
              className="clear-chat-btn"
              onClick={() => {
                if (window.confirm('Clear chat history?')) {
                  window.dispatchEvent(new Event('clear-chat'));
                }
              }}
              title="Clear conversation history"
            >
              <span>🗑️</span>
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="main-content">
        {/* Backend Offline Warning */}
        {isBackendOnline === false && (
          <div className="offline-warning">
            <div className="warning-content">
              <span className="warning-icon">⚠️</span>
              <div>
                <h3>Backend Server Offline</h3>
                <p>Start the backend server to use the chatbot.</p>
                <code>cd Backend && python app.py</code>
              </div>
            </div>
          </div>
        )}

        {/* Chat Interface */}
        <ChatInterface />
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>
          Powered by <span className="text-gradient">VNRVJIET AI</span> •
          Retrieval-Augmented Generation •
          Built with ❤️
        </p>
      </footer>

      {/* Admin Panel Modal */}
      {showAdmin && (
        <AdminPanel onClose={() => setShowAdmin(false)} />
      )}
    </div>
  );
}

export default App;

import React, { useState, useEffect, useCallback } from 'react';
import {
    getIndexStatus,
    getIndexStats,
    getPdfList,
    startIndexing,
    reloadIndex,
    rebuildIndex,
    clearAllData,
} from '../services/api';
import './AdminPanel.css';

/**
 * AdminPanel Component
 * Admin interface for managing the knowledge base
 */
const AdminPanel = ({ onClose }) => {
    const [stats, setStats] = useState(null);
    const [status, setStatus] = useState(null);
    const [pdfs, setPdfs] = useState([]);
    const [targetUrl, setTargetUrl] = useState('');
    const [maxPages, setMaxPages] = useState(100);
    const [isLoading, setIsLoading] = useState(false);
    const [message, setMessage] = useState(null);
    const [activeTab, setActiveTab] = useState('overview');

    // Load data on mount and setup conditional polling
    useEffect(() => {
        loadData();
        loadStatus(); // Initial status check
    }, []);

    // Poll status ONLY when indexing is active
    useEffect(() => {
        if (!status) return; // Wait for initial status load

        const isActiveIndexing = ['scraping', 'processing_pdfs', 'indexing', 'downloading_pdfs'].includes(status.stage);

        if (isActiveIndexing) {
            // Poll every 1.5 seconds during active indexing
            const interval = setInterval(loadStatus, 1500);
            return () => clearInterval(interval);
        }
        // No polling when idle - status only updates on user actions (index, reload)
    }, [status?.stage]);

    const loadData = async () => {
        try {
            const [statsData, statusData, pdfData] = await Promise.all([
                getIndexStats(),
                getIndexStatus(),
                getPdfList(),
            ]);
            setStats(statsData);
            setStatus(statusData);
            setPdfs(pdfData.pdfs || []);
            if (statsData.target_url) {
                setTargetUrl(statsData.target_url);
            }
        } catch (err) {
            console.error('Failed to load admin data:', err);
            showMessage('error', 'Failed to load data. Make sure the backend is running.');
        }
    };

    const loadStatus = async () => {
        try {
            const statusData = await getIndexStatus();

            // Check if indexing just started - auto-switch to index tab
            if (statusData.stage === 'scraping' && status?.stage === 'idle') {
                setActiveTab('index');
            }

            // Check if indexing just completed - show message ONLY ONCE
            // Compare with current status (before update), not previousStatus
            const wasIndexing = status && ['scraping', 'processing_pdfs', 'indexing', 'downloading_pdfs'].includes(status.stage);
            const isNowComplete = statusData.stage === 'complete';

            if (wasIndexing && isNowComplete) {
                loadData();
                showMessage('success', '🎉 Indexing completed successfully!');
            }

            setStatus(statusData);
        } catch (err) {
            console.error('Failed to load status:', err);
        }
    };

    const showMessage = (type, text) => {
        setMessage({ type, text });
        setTimeout(() => setMessage(null), 5000);
    };

    const handleStartIndexing = async () => {
        if (!targetUrl.trim()) {
            showMessage('error', 'Please enter a valid website URL');
            return;
        }

        try {
            setIsLoading(true);
            await startIndexing(targetUrl, maxPages);
            showMessage('success', 'Indexing started! This may take a few minutes...');
            loadStatus();
        } catch (err) {
            showMessage('error', err.response?.data?.detail || 'Failed to start indexing');
        } finally {
            setIsLoading(false);
        }
    };

    const handleReloadIndex = async () => {
        try {
            setIsLoading(true);
            await reloadIndex();
            showMessage('success', 'Index reloaded successfully!');
            loadData();
        } catch (err) {
            showMessage('error', err.response?.data?.detail || 'Failed to reload index');
        } finally {
            setIsLoading(false);
        }
    };

    const handleRebuildIndex = async () => {
        if (!window.confirm('Rebuild index from existing data? This will re-process all pages with text cleaning. No re-crawling needed.')) {
            return;
        }

        try {
            setIsLoading(true);
            await rebuildIndex();
            showMessage('success', 'Index rebuild started! This will take a few minutes...');
            setActiveTab('index');
            loadStatus();
        } catch (err) {
            showMessage('error', err.response?.data?.detail || 'Failed to start rebuild');
        } finally {
            setIsLoading(false);
        }
    };

    const handleClearData = async () => {
        if (!window.confirm('Are you sure you want to delete all data? This cannot be undone.')) {
            return;
        }

        try {
            setIsLoading(true);
            await clearAllData();
            showMessage('success', 'All data cleared successfully');
            loadData();
        } catch (err) {
            showMessage('error', err.response?.data?.detail || 'Failed to clear data');
        } finally {
            setIsLoading(false);
        }
    };

    const isIndexing = status && ['scraping', 'processing_pdfs', 'indexing', 'downloading_pdfs'].includes(status.stage);

    const getProgressPercent = () => {
        if (!status || !status.total || status.total === 0) return 0;
        return Math.min(Math.round((status.current / status.total) * 100), 100);
    };

    // Get stage display info
    const getStageInfo = () => {
        if (!status) return { icon: '⏸️', label: 'Idle', color: '#868e96' };

        const stages = {
            'idle': { icon: '⏸️', label: 'Idle', color: '#868e96' },
            'scraping': { icon: '🔍', label: 'Scraping Website', color: '#339af0' },
            'downloading_pdfs': { icon: '📥', label: 'Downloading PDFs', color: '#7950f2' },
            'processing_pdfs': { icon: '📄', label: 'Processing PDFs', color: '#f59f00' },
            'indexing': { icon: '🧠', label: 'Building Index', color: '#12b886' },
            'complete': { icon: '✅', label: 'Complete', color: '#40c057' },
            'error': { icon: '❌', label: 'Error', color: '#fa5252' }
        };

        return stages[status.stage] || stages['idle'];
    };

    const stageInfo = getStageInfo();

    return (
        <div className="admin-panel-overlay" onClick={onClose}>
            <div className="admin-panel" onClick={(e) => e.stopPropagation()}>
                {/* Header */}
                <div className="admin-header">
                    <div className="admin-header-content">
                        <span className="admin-icon">⚙️</span>
                        <h2 className="admin-title">Admin Control Panel</h2>
                    </div>
                    <button className="close-btn" onClick={onClose}>×</button>
                </div>

                {/* Message Banner */}
                {message && (
                    <div className={`message-banner ${message.type}`}>
                        <span>{message.type === 'success' ? '✓' : '⚠️'}</span>
                        {message.text}
                    </div>
                )}

                {/* Tabs */}
                <div className="admin-tabs">
                    <button
                        className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
                        onClick={() => setActiveTab('overview')}
                    >
                        📊 Overview
                    </button>
                    <button
                        className={`tab-btn ${activeTab === 'index' ? 'active' : ''}`}
                        onClick={() => setActiveTab('index')}
                    >
                        🔄 Index Website
                    </button>
                    <button
                        className={`tab-btn ${activeTab === 'documents' ? 'active' : ''}`}
                        onClick={() => setActiveTab('documents')}
                    >
                        📄 Documents
                    </button>
                </div>

                {/* Tab Content */}
                <div className="admin-content">
                    {/* Overview Tab */}
                    {activeTab === 'overview' && (
                        <div className="tab-content">
                            {/* Show indexing in progress warning */}
                            {isIndexing && (
                                <div className="indexing-warning">
                                    <div className="warning-icon-spin">⏳</div>
                                    <div className="warning-text">
                                        <strong>Indexing in Progress</strong>
                                        <span>{status?.message || 'Processing...'}</span>
                                    </div>
                                    <div className="warning-percent">{getProgressPercent()}%</div>
                                </div>
                            )}

                            <div className="stats-grid">
                                <div className="stat-card">
                                    <div className="stat-icon">📑</div>
                                    <div className="stat-info">
                                        <div className="stat-value">{stats?.pages_scraped || 0}</div>
                                        <div className="stat-label">Pages Indexed</div>
                                    </div>
                                </div>
                                <div className="stat-card">
                                    <div className="stat-icon">📄</div>
                                    <div className="stat-info">
                                        <div className="stat-value">{stats?.pdfs_processed || 0}</div>
                                        <div className="stat-label">PDFs Processed</div>
                                    </div>
                                </div>
                                <div className="stat-card">
                                    <div className="stat-icon">🧩</div>
                                    <div className="stat-info">
                                        <div className="stat-value">{stats?.total_chunks || 0}</div>
                                        <div className="stat-label">Total Chunks</div>
                                    </div>
                                </div>
                                <div className="stat-card">
                                    <div className="stat-icon">{isIndexing ? '⏳' : (stats?.has_index ? '✅' : '❌')}</div>
                                    <div className="stat-info">
                                        <div className="stat-value">
                                            {isIndexing ? 'Indexing...' : (stats?.has_index ? 'Ready' : 'Not Ready')}
                                        </div>
                                        <div className="stat-label">Index Status</div>
                                    </div>
                                </div>
                            </div>

                            {stats?.target_url && (
                                <div className="info-card">
                                    <div className="info-label">Target Website</div>
                                    <div className="info-value">{stats.target_url}</div>
                                </div>
                            )}

                            {stats?.last_updated && (
                                <div className="info-card">
                                    <div className="info-label">Last Updated</div>
                                    <div className="info-value">
                                        {new Date(stats.last_updated).toLocaleString()}
                                    </div>
                                </div>
                            )}

                            <div className="action-buttons">
                                <button
                                    className="btn btn-primary"
                                    onClick={handleRebuildIndex}
                                    disabled={isLoading || isIndexing}
                                    title="Rebuild index from existing data without re-crawling"
                                >
                                    🔨 Rebuild Index
                                </button>
                                <button
                                    className="btn btn-secondary"
                                    onClick={handleReloadIndex}
                                    disabled={isLoading || isIndexing}
                                >
                                    🔄 Reload Index
                                </button>
                                <button
                                    className="btn btn-danger"
                                    onClick={handleClearData}
                                    disabled={isLoading || isIndexing}
                                >
                                    🗑️ Clear All Data
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Index Tab */}
                    {activeTab === 'index' && (
                        <div className="tab-content">
                            {/* Enhanced Progress Section */}
                            {isIndexing && (
                                <div className="progress-section enhanced">
                                    {/* Stage Indicator */}
                                    <div className="stage-indicator">
                                        <div className="stage-icon-wrapper" style={{ '--stage-color': stageInfo.color }}>
                                            <span className="stage-icon">{stageInfo.icon}</span>
                                        </div>
                                        <div className="stage-info">
                                            <span className="stage-label">{stageInfo.label}</span>
                                            <span className="stage-detail">{status?.message}</span>
                                        </div>
                                    </div>

                                    {/* Progress Bar */}
                                    <div className="progress-container">
                                        <div className="progress-header">
                                            <span className="progress-stage">Progress</span>
                                            <span className="progress-percent" style={{ color: stageInfo.color }}>{getProgressPercent()}%</span>
                                        </div>
                                        <div className="progress-bar">
                                            <div
                                                className="progress-fill animated"
                                                style={{
                                                    width: `${getProgressPercent()}%`,
                                                    background: `linear-gradient(90deg, ${stageInfo.color}, ${stageInfo.color}dd)`
                                                }}
                                            />
                                        </div>
                                        <div className="progress-stats">
                                            <span>{status?.current || 0} / {status?.total || 0}</span>
                                            <span>items processed</span>
                                        </div>
                                    </div>

                                    {/* Stage Pipeline */}
                                    <div className="stage-pipeline">
                                        <div className={`pipeline-step ${status?.stage === 'scraping' ? 'active' : status?.stage && ['downloading_pdfs', 'processing_pdfs', 'indexing', 'complete'].includes(status.stage) ? 'done' : ''}`}>
                                            <span className="step-icon">🔍</span>
                                            <span className="step-label">Scrape</span>
                                        </div>
                                        <div className="pipeline-connector"></div>
                                        <div className={`pipeline-step ${status?.stage === 'downloading_pdfs' ? 'active' : status?.stage && ['processing_pdfs', 'indexing', 'complete'].includes(status.stage) ? 'done' : ''}`}>
                                            <span className="step-icon">📥</span>
                                            <span className="step-label">Download</span>
                                        </div>
                                        <div className="pipeline-connector"></div>
                                        <div className={`pipeline-step ${status?.stage === 'processing_pdfs' ? 'active' : status?.stage && ['indexing', 'complete'].includes(status.stage) ? 'done' : ''}`}>
                                            <span className="step-icon">📄</span>
                                            <span className="step-label">Process</span>
                                        </div>
                                        <div className="pipeline-connector"></div>
                                        <div className={`pipeline-step ${status?.stage === 'indexing' ? 'active' : status?.stage === 'complete' ? 'done' : ''}`}>
                                            <span className="step-icon">🧠</span>
                                            <span className="step-label">Index</span>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Completion Message */}
                            {status?.stage === 'complete' && !isIndexing && (
                                <div className="completion-banner">
                                    <div className="completion-icon">🎉</div>
                                    <div className="completion-content">
                                        <h3>Indexing Complete!</h3>
                                        <p>Your knowledge base is ready. You can now ask questions about the indexed content.</p>
                                    </div>
                                </div>
                            )}

                            {/* Index Form */}
                            <div className="form-section">
                                <h3 className="form-title">Index New Website</h3>
                                <p className="form-description">
                                    Enter the college website URL to scrape and index all content including PDF documents.
                                </p>

                                <div className="form-group">
                                    <label className="form-label">Website URL</label>
                                    <input
                                        type="url"
                                        className="input"
                                        placeholder="https://example.edu"
                                        value={targetUrl}
                                        onChange={(e) => setTargetUrl(e.target.value)}
                                        disabled={isLoading || isIndexing}
                                    />
                                </div>

                                <div className="form-group">
                                    <label className="form-label">Maximum Pages</label>
                                    <input
                                        type="number"
                                        className="input"
                                        min="10"
                                        max="1000"
                                        value={maxPages}
                                        onChange={(e) => setMaxPages(parseInt(e.target.value) || 100)}
                                        disabled={isLoading || isIndexing}
                                    />
                                    <p className="form-hint">Recommended: 100-500 pages for typical college websites</p>
                                </div>

                                <button
                                    className="btn btn-primary full-width"
                                    onClick={handleStartIndexing}
                                    disabled={isLoading || isIndexing || !targetUrl.trim()}
                                >
                                    {isIndexing ? '⏳ Indexing in Progress...' : '🚀 Start Indexing'}
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Documents Tab */}
                    {activeTab === 'documents' && (
                        <div className="tab-content">
                            <h3 className="section-title">Indexed PDF Documents ({pdfs.length})</h3>

                            {pdfs.length === 0 ? (
                                <div className="empty-state">
                                    <span className="empty-icon">📂</span>
                                    <p>No PDF documents indexed yet</p>
                                    <p className="empty-hint">Run the indexer to discover and process PDF documents</p>
                                </div>
                            ) : (
                                <div className="pdf-list">
                                    {pdfs.map((pdf, index) => (
                                        <div key={index} className="pdf-item">
                                            <div className="pdf-icon">📄</div>
                                            <div className="pdf-info">
                                                <div className="pdf-name">{pdf.name}</div>
                                                <div className="pdf-meta">
                                                    <span className="pdf-type">{pdf.type}</span>
                                                    <span className="pdf-pages">{pdf.pages} pages</span>
                                                    <span className="pdf-chunks">{pdf.chunks} chunks</span>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default AdminPanel;

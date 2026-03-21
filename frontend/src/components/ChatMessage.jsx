import React from 'react';
import ReactMarkdown from 'react-markdown';
import './ChatMessage.css';

const ChatMessage = ({ message }) => {
    const { role, content, citations, grounded, confidence, error } = message;
    const isBot = role === 'assistant';

    // Filter out FAQ sources from being displayed in the UI
    const displayCitations = citations ? citations.filter(cite =>
        !cite.name.toLowerCase().includes('faq')
    ) : [];

    return (
        <div className={`message ${role} ${error ? 'error' : ''}`}>
            <div className="message-avatar">
                {isBot ? '🤖' : '👤'}
            </div>

            <div className="message-content-wrapper">
                <div className="message-bubble">
                    {/* Render Markdown content */}
                    <div className="markdown-content">
                        <ReactMarkdown>{content}</ReactMarkdown>
                    </div>
                </div>

                {/* Metadata & Footer (Bot Only) */}
                {isBot && !error && (
                    <div className="message-footer">
                        {/* Groundedness Badge */}
                        <div className="status-badges">
                            {grounded !== undefined && (
                                <span className={`badge ${grounded ? 'grounded' : 'not-grounded'}`}
                                    title={grounded ? "Answer based on official docs" : "General knowledge / Low confidence"}>
                                    {grounded ? '✅ Source Grounded' : '⚠️ Low Confidence'}
                                </span>
                            )}
                            {confidence && (
                                <span className="confidence-score" title="Confidence Score">
                                    {(confidence * 100).toFixed(0)}% Match
                                </span>
                            )}
                        </div>

                        {/* Citations / Sources */}
                        {displayCitations.length > 0 && (
                            <div className="citations-area">
                                <p className="citations-header">Sources:</p>
                                <div className="citations-list">
                                    {displayCitations.map((cite, idx) => (
                                        <div key={idx} className="citation-item">
                                            <span className="cite-icon">
                                                {cite.type === 'pdf' ? '📄' : '🌐'}
                                            </span>
                                            <div className="cite-details">
                                                <span className="cite-name">
                                                    {cite.name}
                                                    {cite.page && <span className="cite-page"> (Pg {cite.page})</span>}
                                                </span>
                                                {cite.url && (
                                                    <a href={cite.url} target="_blank" rel="noopener noreferrer" className="cite-link">
                                                        Open Link ↗
                                                    </a>
                                                )}
                                                <p className="cite-snippet">"{cite.snippet}"</p>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default ChatMessage;

import React, { useState, useEffect, useRef } from 'react';
import ChatMessage from './ChatMessage';
import { checkHealth } from '../services/api';
import './ChatInterface.css';

// ── Session key ──────────────────────────────────────────────────────────────
// sessionStorage: persists on F5 reload, clears when tab is closed
const STORAGE_KEY = 'chat_history';

const DEFAULT_MESSAGE = {
    role: 'assistant',
    content: 'Hello! I am your VNRVJIET AI assistant. Ask me anything about the college, admission, fees, or departments.',
    grounded: true
};

const ChatInterface = () => {
    // ── State: load from sessionStorage on first render ──────────────────────
    const [messages, setMessages] = useState(() => {
        try {
            const saved = sessionStorage.getItem(STORAGE_KEY);
            return saved ? JSON.parse(saved) : [DEFAULT_MESSAGE];
        } catch {
            return [DEFAULT_MESSAGE];
        }
    });

    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [suggestions, setSuggestions] = useState([]);
    const messagesEndRef = useRef(null);

    // ── Persist to sessionStorage on every message change ────────────────────
    useEffect(() => {
        try {
            sessionStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
        } catch {
            // sessionStorage full — silently ignore
        }
    }, [messages]);

    // ── Clear chat event (from navbar 🗑️ button) ─────────────────────────────
    useEffect(() => {
        const handleClearChat = () => {
            setMessages([DEFAULT_MESSAGE]);
            sessionStorage.removeItem(STORAGE_KEY);
        };
        window.addEventListener('clear-chat', handleClearChat);
        return () => window.removeEventListener('clear-chat', handleClearChat);
    }, []);

    // ── Auto-scroll ───────────────────────────────────────────────────────────
    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };
    useEffect(scrollToBottom, [messages]);

    // ── Verified suggestion questions ──────────────────────────────────────────
    useEffect(() => {
        setSuggestions([
            "Who is the principal of VNRVJIET?",
            "Who is the HOD of CSE department?",
            "What is the fee structure for B.Tech?",
            "What is the highest placement package in CSE?",
            "What is the average package of CSE?"
        ]);
    }, []);

    // ── Send message ──────────────────────────────────────────────────────────
    const sendMessage = async (text) => {
        if (!text.trim()) return;

        const userMessage = { role: 'user', content: text };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch('http://localhost:5000/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: text })
            });

            if (!response.ok) {
                let errorMessage = "I'm sorry, but I encountered an error processing your question.";
                try {
                    const errorData = await response.json();
                    if (errorData.detail) errorMessage = errorData.detail;
                } catch {
                    errorMessage = `Server error (${response.status}): ${response.status === 503
                        ? 'Service unavailable. Please run indexing first.'
                        : 'Please check backend logs.'
                        }`;
                }
                throw new Error(errorMessage);
            }

            const data = await response.json();
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: data.answer,
                citations: data.citations,
                grounded: data.grounded,
                confidence: data.confidence
            }]);

        } catch (error) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: error.message || "I'm sorry, but I encountered an error connecting to the server. Please check if the backend is running.",
                error: true
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage(input);
        }
    };

    return (
        <div className="chat-interface">
            <div className="chat-messages">
                {messages.map((msg, index) => (
                    <ChatMessage key={index} message={msg} />
                ))}
                {isLoading && (
                    <div className="message assistant loading">
                        <div className="typing-indicator">
                            <span></span><span></span><span></span>
                        </div>
                        <span className="loading-hint">Thinking…</span>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {messages.length === 1 && suggestions?.length > 0 && (
                <div className="suggestions-container">
                    <p className="suggestions-title">Try asking about:</p>
                    <div className="suggestions-grid">
                        {suggestions.map((s, i) => (
                            <button key={i} className="suggestion-chip" onClick={() => sendMessage(s)}>
                                {s}
                            </button>
                        ))}
                    </div>
                </div>
            )}

            <div className="chat-input-area">
                <div className="input-wrapper">
                    <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Ask a question about the college..."
                        rows={1}
                        disabled={isLoading}
                    />
                    <button
                        className="send-btn"
                        onClick={() => sendMessage(input)}
                        disabled={!input.trim() || isLoading}
                    >
                        ➤
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ChatInterface;


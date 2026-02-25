import axios from 'axios';

// API base URL - change this for production
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

// Create axios instance with defaults
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ============== Chat API ==============

/**
 * Send a chat message and get AI response
 * @param {string} query - User's question
 * @param {string} docType - Optional document type filter
 * @returns {Promise} - Chat response with answer and citations
 */
export const sendChatMessage = async (query, docType = null) => {
  const response = await api.post('/api/chat', {
    query,
    doc_type: docType,
  });
  return response.data;
};

/**
 * Get suggested questions
 * @returns {Promise} - List of suggested questions
 */
export const getSuggestions = async () => {
  const response = await api.get('/api/chat/suggestions');
  return response.data;
};

// ============== Admin API ==============

/**
 * Get current indexing status
 * @returns {Promise} - Status object
 */
export const getIndexStatus = async () => {
  const response = await api.get('/api/admin/status');
  return response.data;
};

/**
 * Get index statistics
 * @returns {Promise} - Stats object
 */
export const getIndexStats = async () => {
  const response = await api.get('/api/admin/stats');
  return response.data;
};

/**
 * Get list of indexed PDFs
 * @returns {Promise} - List of PDFs
 */
export const getPdfList = async () => {
  const response = await api.get('/api/admin/pdfs');
  return response.data;
};

/**
 * Start indexing a website
 * @param {string} targetUrl - Website URL to index
 * @param {number} maxPages - Maximum pages to scrape
 * @returns {Promise} - Indexing started confirmation
 */
export const startIndexing = async (targetUrl, maxPages = 100) => {
  const response = await api.post('/api/admin/index', {
    target_url: targetUrl,
    max_pages: maxPages,
  });
  return response.data;
};

/**
 * Reload the knowledge index
 * @returns {Promise} - Reload result
 */
export const reloadIndex = async () => {
  const response = await api.post('/api/admin/reload');
  return response.data;
};

/**
 * Rebuild index from existing data (no re-crawling)
 * @returns {Promise} - Rebuild started confirmation
 */
export const rebuildIndex = async () => {
  const response = await api.post('/api/admin/rebuild');
  return response.data;
};

/**
 * Clear all data and index
 * @returns {Promise} - Clear result
 */
export const clearAllData = async () => {
  const response = await api.delete('/api/admin/clear');
  return response.data;
};

// ============== Document API ==============

/**
 * Get available document types
 * @returns {Promise} - List of document types
 */
export const getDocumentTypes = async () => {
  const response = await api.get('/api/documents/types');
  return response.data;
};

// ============== Health Check ==============

/**
 * Check API health
 * @returns {Promise} - Health status
 */
export const checkHealth = async () => {
  const response = await api.get('/api/health');
  return response.data;
};

export default api;

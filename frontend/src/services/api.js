const configuredBase = import.meta.env.VITE_API_BASE_URL?.trim();
const API_BASE = (configuredBase || '/api').replace(/\/+$/, '');

export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

function friendlyMessage(status) {
  if (status === 404) return 'That service is not available on the server.';
  if (status === 422) return 'Some information is missing or needs correction. Check the form and try again.';
  if (status === 502 || status === 503) return 'This agent or its data provider is currently unavailable.';
  if (status >= 500) return 'The server could not complete this request. Please try again later.';
  return 'The request could not be completed. Check your information and try again.';
}

export async function apiRequest(path, options = {}) {
  const url = `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    console.error('AgroAI API connection failed:', error);
    throw new ApiError('Could not connect to AgroAI. Check that the backend is running.', 0);
  }

  const text = await response.text();
  let data;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }
  if (!response.ok) {
    const detail = data?.detail;
    console.error('AgroAI API error:', response.status, detail || data);
    throw new ApiError(friendlyMessage(response.status), response.status, detail);
  }
  return data;
}

export function getErrorDetail(error) {
  return error?.detail || '';
}

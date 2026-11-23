import { apiRequest } from './api.js';

export function getWeather({ location, latitude, longitude }) {
  const params = new URLSearchParams();
  if (location) params.set('location', location);
  if (latitude !== undefined && latitude !== '') params.set('latitude', latitude);
  if (longitude !== undefined && longitude !== '') params.set('longitude', longitude);
  return apiRequest(`/weather?${params}`);
}

export function detectDisease(image, crop) {
  const form = new FormData();
  form.append('image', image);
  if (crop) form.append('crop', crop);
  return apiRequest('/disease/detect', { method: 'POST', body: form });
}

export function predictYield(request) {
  return apiRequest('/yield/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
}

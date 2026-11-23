import { apiRequest } from './api.js';

export function analyzeFarm(request) {
  return apiRequest('/agro/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
}

export function saveFarmerProfile(profile) {
  return apiRequest('/farmers/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(profile),
  });
}

export function getMarketAnalysis(region, crops) {
  const params = new URLSearchParams();
  crops.forEach((crop) => params.append('crops', crop));
  return apiRequest(`/market-analysis/${encodeURIComponent(region)}?${params}`);
}

export function getRecommendations(farmerId, crops) {
  const query = crops.length ? `?crops=${encodeURIComponent(crops.join(','))}` : '';
  return apiRequest(`/recommendations/${farmerId}${query}`);
}

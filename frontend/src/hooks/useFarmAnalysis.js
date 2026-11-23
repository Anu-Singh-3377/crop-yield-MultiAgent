import { useState } from 'react';
import { analyzeFarm } from '../services/analysisApi.js';

export function useFarmAnalysis() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function runAnalysis(request) {
    setLoading(true);
    setError('');
    try {
      const response = await analyzeFarm(request);
      setResult(response);
      return response;
    } catch (requestError) {
      setError(requestError.message || 'The farm assessment could not be completed.');
      throw requestError;
    } finally {
      setLoading(false);
    }
  }

  return { result, loading, error, runAnalysis, setResult };
}

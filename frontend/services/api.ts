// frontend/services/api.ts

/**
 * API Service for ARM-GAN Platform
 * Communicates strictly with the stable /api/v1 REST API.
 */

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function fetchWithHandler(endpoint: string, options?: RequestInit) {
  const url = `${BASE_URL}${endpoint}`;
  
  const defaultHeaders = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  };

  const response = await fetch(url, {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new ApiError(
      errorData?.message || `API Error: ${response.status} ${response.statusText}`, 
      response.status
    );
  }

  return response.json();
}

export const api = {
  getSystemHealth: () => fetchWithHandler('/health'),
  
  getStudies: (params?: Record<string, string>) => {
    const query = new URLSearchParams(params).toString();
    return fetchWithHandler(`/studies${query ? `?${query}` : ''}`);
  },

  getStudy: (id: string) => fetchWithHandler(`/studies/${id}`),

  uploadStudy: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${BASE_URL}/studies/upload`, {
      method: 'POST',
      body: formData,
      // Do not set Content-Type, let browser set it with boundary
    });

    if (!response.ok) {
      throw new ApiError(`Upload failed: ${response.statusText}`, response.status);
    }
    
    return response.json();
  },

  getInferenceStatus: (id: string) => fetchWithHandler(`/inference/${id}/status`),
};

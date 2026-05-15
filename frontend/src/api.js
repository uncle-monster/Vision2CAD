const API_BASE = import.meta.env.VITE_API_BASE || '';

export async function uploadImage(file, category) {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('category', category);

  const res = await fetch(`${API_BASE}/api/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error || `Upload failed (${res.status})`);
  }
  return res.json();
}

export async function getJobStatus(jobId) {
  const res = await fetch(`${API_BASE}/api/status/${jobId}`);
  if (!res.ok) {
    if (res.status === 404) return null;
    const err = await res.json();
    throw new Error(err.error || `Status check failed (${res.status})`);
  }
  return res.json();
}

export function getPreviewUrl(jobId) {
  return `${API_BASE}/api/preview/${jobId}`;
}

export function getDownloadUrl(jobId, type) {
  return `${API_BASE}/api/download/${jobId}/${type}`;
}

export async function getCategories() {
  const res = await fetch(`${API_BASE}/api/categories`);
  if (!res.ok) {
    throw new Error(`Failed to fetch categories (${res.status})`);
  }
  return res.json();
}

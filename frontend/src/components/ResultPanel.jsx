function formatSize(bytes) {
  if (!bytes || bytes === 0) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export default function ResultPanel({ jobId, objSize, stepSize, onReset }) {
  const previewUrl = `/api/preview/${jobId}`;
  const objUrl = `/api/download/${jobId}/obj`;
  const stepUrl = `/api/download/${jobId}/step`;

  return (
    <div className="result-panel">
      <div className="result-preview-card">
        <div className="result-preview-header">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22C55E" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
            <polyline points="22 4 12 14.01 9 11.01" />
          </svg>
          <span>生成完成</span>
        </div>
        <img
          src={previewUrl}
          alt="CAD model preview"
          className="result-preview-img"
          loading="lazy"
        />
      </div>

      <div className="result-downloads">
        <a href={objUrl} download className="download-btn obj-btn">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="16" y1="13" x2="8" y2="13" />
            <line x1="16" y1="17" x2="8" y2="17" />
            <polyline points="10 9 9 9 8 9" />
          </svg>
          <span>
            <strong>下载 OBJ</strong>
            <small>3D 网格模型{objSize ? ` (${formatSize(objSize)})` : ''}</small>
          </span>
        </a>

        <a href={stepUrl} download className="download-btn step-btn">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
            <line x1="3" y1="9" x2="21" y2="9" />
            <line x1="9" y1="21" x2="9" y2="9" />
          </svg>
          <span>
            <strong>下载 STEP</strong>
            <small>可编辑 CAD 实体{stepSize ? ` (${formatSize(stepSize)})` : ''}</small>
          </span>
        </a>
      </div>

      <button type="button" className="reset-btn" onClick={onReset}>
        开始新的转换
      </button>
    </div>
  );
}

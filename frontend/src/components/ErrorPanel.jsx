export default function ErrorPanel({ message, onRetry, onReset }) {
  return (
    <div className="error-panel" role="alert">
      <div className="error-icon-wrap">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#EF4444" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
      </div>
      <h3 className="error-title">处理失败</h3>
      <p className="error-message">{message || '发生未知错误，请重试'}</p>
      <div className="error-actions">
        {onRetry && (
          <button type="button" className="retry-btn" onClick={onRetry}>
            重试
          </button>
        )}
        <button type="button" className="reset-btn-outline" onClick={onReset}>
          重新开始
        </button>
      </div>
    </div>
  );
}

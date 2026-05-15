import { useState, useEffect } from 'react';

const STAGES = [
  { key: 'preprocessing', label: '预处理', sub: '背景移除' },
  { key: 'stage1', label: 'Stage 1', sub: 'VLM 结构生成' },
  { key: 'stage2', label: 'Stage 2', sub: 'GMFlow 参数预测' },
  { key: 'done', label: '完成', sub: 'CAD 实体生成' },
];

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

export default function ProgressPanel({ progress }) {
  const [elapsedDisplay, setElapsedDisplay] = useState('00:00');
  const { stage, progress: pct, message, elapsed_seconds, stage_times } = progress;

  // Animate elapsed time
  useEffect(() => {
    if (elapsed_seconds == null) return;
    const startTime = Date.now() - elapsed_seconds * 1000;
    const tick = () => setElapsedDisplay(formatTime((Date.now() - startTime) / 1000));
    tick();
    const interval = setInterval(tick, 1000);
    return () => clearInterval(interval);
  }, [elapsed_seconds]);

  // Determine which stage is active
  const currentStageIdx = STAGES.findIndex((s) => s.key === stage);
  const doneStages = STAGES.filter((_, i) => i < currentStageIdx);
  const isDone = stage === 'done' || progress.status === 'done';

  return (
    <div className="progress-panel">
      {/* Stage pipeline */}
      <div className="stage-pipeline">
        {STAGES.map((s, i) => {
          const isActive = i === currentStageIdx && !isDone;
          const isCompleted = i < currentStageIdx || (i === currentStageIdx && isDone);
          return (
            <div key={s.key} className="stage-item">
              <div className={`stage-dot ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}>
                {isCompleted ? (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                ) : isActive ? (
                  <div className="stage-pulse" />
                ) : (
                  <span className="stage-num">{i + 1}</span>
                )}
              </div>
              <div className="stage-info">
                <span className="stage-label">{s.label}</span>
                <span className="stage-sub">{s.sub}</span>
              </div>
              {i < STAGES.length - 1 && (
                <div className={`stage-line ${isCompleted ? 'completed' : ''}`} />
              )}
            </div>
          );
        })}
      </div>

      {/* Progress bar */}
      <div className="progress-bar-wrap">
        <div className="progress-bar-track">
          <div
            className="progress-bar-fill"
            style={{ width: `${Math.min(pct || 0, 100)}%` }}
          />
        </div>
        <span className="progress-pct">{Math.round(pct || 0)}%</span>
      </div>

      {/* Status row */}
      <div className="progress-status-row">
        <span className="progress-message">{message || '准备中...'}</span>
        <span className="progress-elapsed">{elapsedDisplay}</span>
      </div>
    </div>
  );
}

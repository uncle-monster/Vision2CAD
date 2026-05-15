import { useState, useCallback, useEffect } from 'react';
import Header from './components/Header';
import UploadZone from './components/UploadZone';
import CategorySelector from './components/CategorySelector';
import ProgressPanel from './components/ProgressPanel';
import ResultPanel from './components/ResultPanel';
import ErrorPanel from './components/ErrorPanel';
import useJobProgress from './useJobProgress';
import { uploadImage, getCategories } from './api';
import './App.css';

const DEFAULT_CATEGORIES = [
  { id: 'chair', label: '椅子' },
  { id: 'table', label: '桌子' },
  { id: 'storagefurniture', label: '柜子' },
];

export default function App() {
  // App state: idle | uploaded | processing | done | error
  const [appState, setAppState] = useState('idle');
  const [categories, setCategories] = useState(DEFAULT_CATEGORIES);
  const [category, setCategory] = useState(DEFAULT_CATEGORIES[0].id);
  const [file, setFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [resultSizes, setResultSizes] = useState({ objSize: 0, stepSize: 0 });

  const progress = useJobProgress(appState === 'processing' ? jobId : null);

  // Fetch categories on mount
  useEffect(() => {
    getCategories()
      .then((data) => setCategories(data.categories))
      .catch(() => {}); // fall back to defaults
  }, []);

  // Handle "done" or "error" from WebSocket
  useEffect(() => {
    if (progress.status === 'done') {
      setAppState('done');
      setResultSizes({ objSize: progress.obj_size, stepSize: progress.step_size });
    } else if (progress.status === 'error') {
      setAppState('error');
      setErrorMessage(progress.message || '处理失败');
    }
  }, [progress.status, progress.obj_size, progress.step_size, progress.message]);

  const handleFileSelect = useCallback((f) => {
    setFile(f);
    if (f) {
      setAppState('uploaded');
      setErrorMessage('');
    } else {
      setAppState('idle');
    }
  }, []);

  const handleCategorySelect = useCallback((id) => {
    setCategory(id);
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!file) return;
    setAppState('processing');
    setErrorMessage('');
    try {
      const result = await uploadImage(file, category);
      setJobId(result.job_id);
    } catch (err) {
      setAppState('error');
      setErrorMessage(err.message || '上传失败');
    }
  }, [file, category]);

  const handleRetry = useCallback(() => {
    handleGenerate();
  }, [handleGenerate]);

  const handleReset = useCallback(() => {
    setAppState('idle');
    setFile(null);
    setJobId(null);
    setErrorMessage('');
    setResultSizes({ objSize: 0, stepSize: 0 });
  }, []);

  return (
    <div className="app">
      <div className="app-container">
        <Header />

        <main className="app-main">
          {/* Idle / Uploaded: Show upload & category */}
          {(appState === 'idle' || appState === 'uploaded') && (
            <>
              <CategorySelector
                categories={categories}
                selected={category}
                onSelect={handleCategorySelect}
              />
              <UploadZone onFileSelect={handleFileSelect} selectedFile={file} />
              {appState === 'uploaded' && (
                <button type="button" className="generate-btn" onClick={handleGenerate}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  开始生成 CAD 模型
                </button>
              )}
            </>
          )}

          {/* Processing: Show progress */}
          {appState === 'processing' && (
            <ProgressPanel progress={progress} />
          )}

          {/* Done: Show results */}
          {appState === 'done' && (
            <ResultPanel
              jobId={jobId}
              objSize={resultSizes.objSize}
              stepSize={resultSizes.stepSize}
              onReset={handleReset}
            />
          )}

          {/* Error */}
          {appState === 'error' && (
            <ErrorPanel
              message={errorMessage}
              onRetry={appState === 'error' ? handleRetry : null}
              onReset={handleReset}
            />
          )}
        </main>

        <footer className="app-footer">
          <span>Powered by Llama 3.2 + GMFlow</span>
        </footer>
      </div>
    </div>
  );
}

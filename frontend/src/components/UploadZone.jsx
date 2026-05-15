import { useRef, useState, useCallback } from 'react';

export default function UploadZone({ onFileSelect, selectedFile }) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragIn = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(true);
  }, []);

  const handleDragOut = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      e.stopPropagation();
      setDragOver(false);
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
          onFileSelect(file);
        }
      }
    },
    [onFileSelect]
  );

  const handleChange = useCallback(
    (e) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        onFileSelect(files[0]);
      }
    },
    [onFileSelect]
  );

  const handleClick = () => {
    inputRef.current?.click();
  };

  return (
    <div
      className={`upload-zone ${dragOver ? 'drag-over' : ''} ${selectedFile ? 'has-file' : ''}`}
      onClick={!selectedFile ? handleClick : undefined}
      onDragEnter={handleDragIn}
      onDragLeave={handleDragOut}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleClick(); }}
      aria-label="Upload image"
    >
      <input
        ref={inputRef}
        type="file"
        accept=".jpg,.jpeg,.png"
        onChange={handleChange}
        className="upload-input-hidden"
        aria-hidden="true"
      />

      {selectedFile ? (
        <div className="upload-preview">
          <img
            src={URL.createObjectURL(selectedFile)}
            alt="Preview"
            className="upload-thumbnail"
          />
          <div className="upload-file-info">
            <span className="upload-filename">{selectedFile.name}</span>
            <span className="upload-filesize">
              {(selectedFile.size / 1024).toFixed(1)} KB
            </span>
          </div>
          <button
            type="button"
            className="upload-change-btn"
            onClick={(e) => {
              e.stopPropagation();
              onFileSelect(null);
            }}
          >
            更换
          </button>
        </div>
      ) : (
        <div className="upload-placeholder">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <p className="upload-text">拖拽图片到此处，或点击选择</p>
          <p className="upload-hint">支持 JPG、PNG 格式，最大 20MB</p>
        </div>
      )}
    </div>
  );
}

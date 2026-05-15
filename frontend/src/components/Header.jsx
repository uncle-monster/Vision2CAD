export default function Header() {
  return (
    <header className="app-header">
      <div className="header-logo">
        <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
          <rect width="32" height="32" rx="8" fill="#22C55E" fillOpacity="0.15" />
          <path d="M8 20V12L12 8L16 12L20 8L24 12V20L16 24L8 20Z" stroke="#22C55E" strokeWidth="1.5" strokeLinejoin="round" />
          <circle cx="16" cy="15" r="3" stroke="#22C55E" strokeWidth="1.5" />
        </svg>
        <h1>Img2CAD</h1>
      </div>
      <p className="header-subtitle">AI 驱动的图片转 CAD 模型工具</p>
    </header>
  );
}

const ICONS = {
  chair: (
    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M6 3v12M6 15v3a2 2 0 0 0 2 2h2v-5M18 3v12M18 15v3a2 2 0 0 1-2 2h-2v-5" />
      <rect x="5" y="3" width="3" height="2" rx="0.5" />
      <rect x="16" y="3" width="3" height="2" rx="0.5" />
      <path d="M8 10h8v5H8z" />
      <path d="M10 3h4v7h-4z" />
    </svg>
  ),
  table: (
    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="9" width="18" height="2" rx="0.5" />
      <rect x="3" y="17" width="18" height="2" rx="0.5" />
      <path d="M6 11v6M12 11v6M18 11v6" />
      <rect x="5" y="3" width="14" height="2" rx="1" />
    </svg>
  ),
  storagefurniture: (
    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <line x1="12" y1="3" x2="12" y2="21" />
      <line x1="3" y1="9" x2="21" y2="9" />
      <line x1="3" y1="15" x2="21" y2="15" />
      <circle cx="8" cy="6" r="0.8" fill="currentColor" />
      <circle cx="16" cy="6" r="0.8" fill="currentColor" />
    </svg>
  ),
};

export default function CategorySelector({ categories, selected, onSelect }) {
  return (
    <div className="category-selector">
      <label className="category-label">选择物体类别</label>
      <div className="category-cards">
        {categories.map((cat) => (
          <button
            key={cat.id}
            type="button"
            className={`category-card ${selected === cat.id ? 'selected' : ''}`}
            onClick={() => onSelect(cat.id)}
            aria-pressed={selected === cat.id}
          >
            <div className="category-icon">{ICONS[cat.id]}</div>
            <span className="category-name">{cat.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

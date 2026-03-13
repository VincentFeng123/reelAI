const LOADING_LETTERS = ["R", "E", "E", "L", "A", "I"] as const;

export function FullscreenLoadingScreen() {
  return (
    <div className="loading" aria-busy="true">
      <div className="loading-text" role="status" aria-live="polite" aria-label="Loading ReelAI">
        {LOADING_LETTERS.map((letter, index) => (
          <span key={`${letter}-${index}`} className="loading-text-words">
            {letter}
          </span>
        ))}
      </div>
    </div>
  );
}

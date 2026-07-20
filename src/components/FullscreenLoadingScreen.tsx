const LOADING_LETTERS = ["R", "E", "E", "L", "A", "I"] as const;

export function FullscreenLoadingScreen() {
  return (
    <div
      className="reelai-loading-screen"
      aria-busy="true"
    >
      <div
        className="reelai-loading-wordmark"
        role="status"
        aria-live="polite"
        aria-label="Loading ReelAI"
      >
        {LOADING_LETTERS.map((letter, index) => (
          <span
            key={`${letter}-${index}`}
            className="reelai-loading-letter"
            style={{ animationDelay: `${index * 0.2}s` }}
            aria-hidden="true"
          >
            {letter}
          </span>
        ))}
      </div>
    </div>
  );
}

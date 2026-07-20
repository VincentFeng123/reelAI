export function FullscreenLoadingScreen() {
  return (
    <div
      className="fixed inset-0 grid min-h-[100dvh] place-items-center bg-black text-white"
      aria-busy="true"
    >
      <div
        className="text-lg font-semibold tracking-[-0.02em]"
        role="status"
        aria-live="polite"
        aria-label="Loading ReelAI"
      >
        ReelAI
      </div>
    </div>
  );
}

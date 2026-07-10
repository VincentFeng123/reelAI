import { Settings } from "../types";

export function SettingsDrawer({
  open,
  settings,
  onChange,
  onClose,
}: {
  open: boolean;
  settings: Settings;
  onChange: (s: Settings) => void;
  onClose: () => void;
}) {
  return (
    <>
      <div
        className={`fixed inset-0 z-40 bg-black/50 backdrop-blur-sm transition-opacity duration-300 ${
          open ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
        onClick={onClose}
      />
      <aside
        className={`fixed right-0 top-0 z-50 h-full w-full sm:max-w-sm glass rounded-none sm:rounded-l-2xl
          p-6 overflow-y-auto transition-transform duration-300 ${
            open ? "translate-x-0" : "translate-x-full"
          }`}
      >
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Settings</h2>
          <button className="btn-ghost px-3 py-1.5" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        <div className="mt-6 space-y-6">
          <div>
            <label className="block text-xs font-medium text-white/60 mb-2">Engines</label>
            <div className="flex flex-wrap gap-2">
              <span className="tag normal-case">Transcript: Supadata</span>
              <span className="tag normal-case !bg-cyan-500/15 !text-cyan-200 !ring-cyan-400/30">
                Select: Gemini
              </span>
            </div>
            <p className="mt-1.5 text-xs text-white/40">
              Clips play from YouTube at full quality; nothing is downloaded until you export.
            </p>
          </div>

          <div>
            <label className="block text-xs font-medium text-white/60 mb-2">Export resolution</label>
            <div className="grid grid-cols-4 gap-2">
              {[720, 1080, 1440, 2160].map((r) => (
                <button
                  key={r}
                  onClick={() => onChange({ ...settings, export_resolution: r })}
                  className={`btn-ghost text-sm ${
                    settings.export_resolution === r ? "ring-2 ring-violet-400/60" : ""
                  }`}
                >
                  {r === 2160 ? "4K" : `${r}p`}
                </button>
              ))}
            </div>
            <p className="mt-1.5 text-xs text-white/40">Used only when you export a clip to .mp4.</p>
          </div>

          <div>
            <label className="block text-xs font-medium text-white/60 mb-2">Clip engine</label>
            <select
              className="field"
              value={settings.clip_engine ?? "gemini"}
              onChange={(e) => onChange({ ...settings, clip_engine: e.target.value })}
            >
              <option value="gemini">Gemini one-pass (default)</option>
              <option value="topic">Full/topic (experimental)</option>
              <option value="unit">Full/unit (experimental)</option>
            </select>
            <p className="mt-1.5 text-xs text-white/40">
              One-pass uses Supadata captions only. The experimental engines run the heavier
              structure pipeline.
            </p>
          </div>

          <label className="flex items-center justify-between gap-3 cursor-pointer">
            <span className="text-sm text-white/70">
              Allow clips to end on “?” or “!”
              <span className="block text-xs text-white/40">Default: end on a period.</span>
            </span>
            <input
              type="checkbox"
              className="h-5 w-5 accent-violet-500"
              checked={settings.allow_question_exclaim_ends}
              onChange={(e) =>
                onChange({ ...settings, allow_question_exclaim_ends: e.target.checked })
              }
            />
          </label>

          <div>
            <label className="block text-xs font-medium text-white/60 mb-2">
              Variety ↔ Relevance ({settings.mmr_lambda.toFixed(2)})
            </label>
            <input
              type="range"
              min={0.3}
              max={0.9}
              step={0.05}
              value={settings.mmr_lambda}
              onChange={(e) => onChange({ ...settings, mmr_lambda: parseFloat(e.target.value) })}
              className="w-full accent-violet-500"
            />
            <p className="mt-1 text-xs text-white/40">
              Lower = more distinct facets, higher = stay tighter to the topic.
            </p>
          </div>
        </div>
      </aside>
    </>
  );
}

"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

type GenerationMode = "slow" | "fast";
type SearchInputMode = "topic" | "source" | "file";

type SettingsPanelProps = {
  onClearSearchData: () => void;
};

type SavedPreferences = {
  generationMode: GenerationMode;
  defaultInputMode: SearchInputMode;
  minRelevanceThreshold: number;
  startMuted: boolean;
};

const GENERATION_MODE_STORAGE_KEY = "studyreels-generation-mode";
const SEARCH_INPUT_MODE_STORAGE_KEY = "studyreels-search-input-mode";
const MIN_RELEVANCE_STORAGE_KEY = "studyreels-min-relevance-threshold";
const MUTED_STORAGE_KEY = "studyreels-muted";
const COMMUNITY_SETS_STORAGE_KEY = "studyreels-community-sets";
const DEFAULT_MIN_RELEVANCE = 0.0;
const MIN_RELEVANCE = 0.0;
const MAX_RELEVANCE = 0.6;
const RELEVANCE_STEP = 0.02;

const SEARCH_INPUT_OPTIONS: Array<{ value: SearchInputMode; label: string }> = [
  { value: "source", label: "Text" },
  { value: "topic", label: "Topic" },
  { value: "file", label: "File" },
];

export function SettingsPanel({ onClearSearchData }: SettingsPanelProps) {
  const [generationMode, setGenerationMode] = useState<GenerationMode>("slow");
  const [defaultInputMode, setDefaultInputMode] = useState<SearchInputMode>("source");
  const [minRelevanceThreshold, setMinRelevanceThreshold] = useState(DEFAULT_MIN_RELEVANCE);
  const [startMuted, setStartMuted] = useState(true);
  const [savedPreferences, setSavedPreferences] = useState<SavedPreferences | null>(null);
  const [settingsHydrated, setSettingsHydrated] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    let nextGenerationMode: GenerationMode = "slow";
    const savedMode = window.localStorage.getItem(GENERATION_MODE_STORAGE_KEY);
    if (savedMode === "slow" || savedMode === "fast") {
      nextGenerationMode = savedMode;
    }

    let nextInputMode: SearchInputMode = "source";
    const savedInputMode = window.localStorage.getItem(SEARCH_INPUT_MODE_STORAGE_KEY);
    if (savedInputMode === "topic" || savedInputMode === "source" || savedInputMode === "file") {
      nextInputMode = savedInputMode;
    }

    let nextMinRelevance = DEFAULT_MIN_RELEVANCE;
    const savedThreshold = Number(window.localStorage.getItem(MIN_RELEVANCE_STORAGE_KEY));
    if (Number.isFinite(savedThreshold)) {
      nextMinRelevance = Math.max(MIN_RELEVANCE, Math.min(MAX_RELEVANCE, savedThreshold));
    }

    let nextStartMuted = true;
    const savedMuted = window.localStorage.getItem(MUTED_STORAGE_KEY);
    if (savedMuted === "0") {
      nextStartMuted = false;
    } else if (savedMuted === "1") {
      nextStartMuted = true;
    }

    setGenerationMode(nextGenerationMode);
    setDefaultInputMode(nextInputMode);
    setMinRelevanceThreshold(nextMinRelevance);
    setStartMuted(nextStartMuted);
    setSavedPreferences({
      generationMode: nextGenerationMode,
      defaultInputMode: nextInputMode,
      minRelevanceThreshold: Number(nextMinRelevance.toFixed(2)),
      startMuted: nextStartMuted,
    });
    setSettingsHydrated(true);
  }, []);

  const showNotice = useCallback((message: string) => {
    setNotice(message);
    if (typeof window === "undefined") {
      return;
    }
    window.setTimeout(() => {
      setNotice((current) => (current === message ? null : current));
    }, 2000);
  }, []);

  const resetPreferences = () => {
    setGenerationMode("slow");
    setDefaultInputMode("source");
    setMinRelevanceThreshold(DEFAULT_MIN_RELEVANCE);
    setStartMuted(true);
    showNotice("Defaults loaded. Save to apply.");
  };

  const clearCommunitySetsCache = () => {
    if (typeof window !== "undefined") {
      window.localStorage.removeItem(COMMUNITY_SETS_STORAGE_KEY);
    }
    showNotice("Saved community set cache cleared.");
  };

  const settingsSummary = useMemo(() => {
    const speedLabel = generationMode === "fast" ? "Fast" : "Slow";
    const inputLabel = SEARCH_INPUT_OPTIONS.find((option) => option.value === defaultInputMode)?.label ?? "Text";
    return `${speedLabel} mode · ${inputLabel} input · Match ${minRelevanceThreshold.toFixed(2)}+ · ${startMuted ? "Muted" : "Sound on"}`;
  }, [defaultInputMode, generationMode, minRelevanceThreshold, startMuted]);

  const hasUnsavedChanges = useMemo(() => {
    if (!savedPreferences) {
      return false;
    }
    const currentMinRelevance = Number(minRelevanceThreshold.toFixed(2));
    return (
      savedPreferences.generationMode !== generationMode
      || savedPreferences.defaultInputMode !== defaultInputMode
      || savedPreferences.minRelevanceThreshold !== currentMinRelevance
      || savedPreferences.startMuted !== startMuted
    );
  }, [defaultInputMode, generationMode, minRelevanceThreshold, savedPreferences, startMuted]);

  const savePreferences = () => {
    if (typeof window === "undefined" || !settingsHydrated) {
      return;
    }
    const normalizedMinRelevance = Number(minRelevanceThreshold.toFixed(2));
    window.localStorage.setItem(GENERATION_MODE_STORAGE_KEY, generationMode);
    window.localStorage.setItem(SEARCH_INPUT_MODE_STORAGE_KEY, defaultInputMode);
    window.localStorage.setItem(MIN_RELEVANCE_STORAGE_KEY, normalizedMinRelevance.toFixed(2));
    window.localStorage.setItem(MUTED_STORAGE_KEY, startMuted ? "1" : "0");
    setSavedPreferences({
      generationMode,
      defaultInputMode,
      minRelevanceThreshold: normalizedMinRelevance,
      startMuted,
    });
    showNotice("Settings saved.");
  };

  return (
    <div className="flex h-full min-h-0 w-full justify-center overflow-y-auto px-6 pt-6 pb-8 md:px-10 md:pt-8 md:pb-10 lg:px-10">
      <div className="w-full max-w-[980px]">
        <header className="mb-6 md:mb-8">
          <h1 className="text-3xl font-semibold tracking-tight text-white md:text-4xl">Settings</h1>
          <p className="mt-2 text-sm text-white/70">Configure your default search behavior and quickly manage saved app data.</p>
        </header>

        <div className="rounded-3xl bg-white/[0.07] p-4 backdrop-blur-[2px] md:p-6">
          <p className="text-[11px] font-semibold uppercase tracking-[0.11em] text-white/62">Search Defaults</p>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div>
              <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.11em] text-white/62">Generation Speed</label>
              <div className="relative grid h-12 grid-cols-2 items-center rounded-2xl border border-white/20 bg-white/[0.08] p-1 backdrop-blur-[2px]">
                <span
                  aria-hidden="true"
                  className={`pointer-events-none absolute bottom-1 left-1 top-1 w-[calc(50%-4px)] rounded-xl bg-white transition-transform duration-300 ease-out ${
                    generationMode === "fast" ? "translate-x-full" : "translate-x-0"
                  }`}
                />
                <button
                  type="button"
                  onClick={() => setGenerationMode("slow")}
                  className={`relative z-10 rounded-xl px-2 py-2 text-[11px] font-semibold uppercase tracking-[0.05em] transition-colors duration-200 ${
                    generationMode === "slow" ? "text-black" : "text-white/75 hover:text-white"
                  }`}
                  aria-pressed={generationMode === "slow"}
                >
                  Slow
                </button>
                <button
                  type="button"
                  onClick={() => setGenerationMode("fast")}
                  className={`relative z-10 rounded-xl px-2 py-2 text-[11px] font-semibold uppercase tracking-[0.05em] transition-colors duration-200 ${
                    generationMode === "fast" ? "text-black" : "text-white/75 hover:text-white"
                  }`}
                  aria-pressed={generationMode === "fast"}
                >
                  Fast
                </button>
              </div>
            </div>

            <div>
              <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.11em] text-white/62">Default Input Mode</label>
              <div className="relative grid h-12 grid-cols-3 items-center rounded-2xl border border-white/20 bg-white/[0.08] p-1 backdrop-blur-[2px]">
                <span
                  aria-hidden="true"
                  className="pointer-events-none absolute bottom-1 left-1 top-1 w-[calc((100%-8px)/3)] rounded-xl bg-white transition-transform duration-300 ease-out"
                  style={{
                    transform: `translateX(${SEARCH_INPUT_OPTIONS.findIndex((option) => option.value === defaultInputMode) * 100}%)`,
                  }}
                />
                {SEARCH_INPUT_OPTIONS.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => setDefaultInputMode(option.value)}
                    className={`relative z-10 rounded-xl px-2 py-2 text-[11px] font-semibold uppercase tracking-[0.05em] transition-colors duration-200 ${
                      defaultInputMode === option.value ? "text-black" : "text-white/75 hover:text-white"
                    }`}
                    aria-pressed={defaultInputMode === option.value}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-4 rounded-2xl bg-white/[0.06] p-3.5 backdrop-blur-[2px] md:p-4">
            <div>
              <div className="flex items-center justify-between gap-3">
                <p className="text-sm font-semibold text-white/95">Similarity / matching threshold</p>
                <span className="rounded-md bg-black/35 px-2 py-0.5 text-xs font-semibold text-white/90">
                  {minRelevanceThreshold.toFixed(2)}+
                </span>
              </div>
              <p className="mt-1 text-xs text-white/62">
                Higher values keep results tightly related and filter out unrelated content.
              </p>
              <input
                type="range"
                min={MIN_RELEVANCE}
                max={MAX_RELEVANCE}
                step={RELEVANCE_STEP}
                value={minRelevanceThreshold}
                onChange={(event) => {
                  const value = Number(event.target.value);
                  if (!Number.isFinite(value)) {
                    return;
                  }
                  setMinRelevanceThreshold(Math.max(MIN_RELEVANCE, Math.min(MAX_RELEVANCE, value)));
                }}
                className="mt-3 h-2 w-full cursor-pointer appearance-none rounded-full bg-white/18 accent-white"
              />
              <div className="mt-1.5 flex items-center justify-between text-[11px] text-white/52">
                <span>Loose</span>
                <span>Balanced</span>
                <span>Strict</span>
              </div>
            </div>
          </div>

          <div className="mt-4 rounded-2xl bg-white/[0.06] p-3.5 backdrop-blur-[2px] md:p-4">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm font-semibold text-white/95">Start reels muted</p>
                <p className="mt-1 text-xs text-white/62">Controls the default audio state when opening the feed.</p>
              </div>
              <button
                type="button"
                onClick={() => setStartMuted((prev) => !prev)}
                aria-pressed={startMuted}
                className={`relative inline-flex h-7 w-12 shrink-0 rounded-full border transition-colors duration-200 ${
                  startMuted ? "border-white bg-white" : "border-white/32 bg-black/50"
                }`}
              >
                <span
                  className={`absolute top-0.5 h-[22px] w-[22px] rounded-full transition-transform duration-200 ${
                    startMuted ? "translate-x-[24px] bg-black" : "translate-x-[2px] bg-white"
                  }`}
                />
              </button>
            </div>
          </div>

          <p className="mt-4 text-xs text-white/52">Current defaults: {settingsSummary}</p>
        </div>

        <div className="mt-4 rounded-3xl bg-white/[0.07] p-4 backdrop-blur-[2px] md:mt-5 md:p-6">
          <p className="text-[11px] font-semibold uppercase tracking-[0.11em] text-white/62">Utilities</p>
          <p className="mt-2 text-xs text-white/62">Useful maintenance actions for search history and local cache.</p>

          <div className="mt-4 grid gap-2 md:grid-cols-3">
            <button
              type="button"
              onClick={() => {
                onClearSearchData();
                showNotice("Search history and session data cleared.");
              }}
              className="rounded-xl border border-white/20 bg-black/36 px-3 py-2 text-xs font-semibold text-white/86 transition hover:bg-white/10"
            >
              Clear search data
            </button>
            <button
              type="button"
              onClick={clearCommunitySetsCache}
              className="rounded-xl border border-white/20 bg-black/36 px-3 py-2 text-xs font-semibold text-white/86 transition hover:bg-white/10"
            >
              Clear set cache
            </button>
            <button
              type="button"
              onClick={resetPreferences}
              className="rounded-xl border border-white/20 bg-black/36 px-3 py-2 text-xs font-semibold text-white/86 transition hover:bg-white/10"
            >
              Reset defaults
            </button>
          </div>

        </div>

        <div className="mt-2 flex items-end justify-between gap-3">
          <p className="min-h-5 text-left text-xs text-white/72">{notice ?? ""}</p>
          <button
            type="button"
            onClick={savePreferences}
            disabled={!settingsHydrated || !hasUnsavedChanges}
            className="inline-flex min-w-[10rem] items-center justify-center whitespace-nowrap rounded-xl border border-white/24 bg-white px-7 py-3 text-sm font-semibold text-black transition-colors hover:bg-white/90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}

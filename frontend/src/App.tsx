import { useMemo, useState } from "react";
import { createJob } from "./api";
import { useJobStream } from "./hooks/useJobStream";
import { Phase, Settings } from "./types";
import { InputCard } from "./components/InputCard";
import { ProcessingStepper } from "./components/ProcessingStepper";
import { ResultsGrid } from "./components/ResultsGrid";
import { SettingsDrawer } from "./components/SettingsDrawer";

const DEFAULT_SETTINGS: Settings = {
  allow_question_exclaim_ends: false,
  mmr_lambda: 0.6,
  export_resolution: 1080,
};

export default function App() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const snap = useJobStream(jobId);

  const phase: Phase = useMemo(() => {
    if (submitError) return "error";
    if (!jobId) return "input";
    if (!snap) return "processing";
    if (snap.status === "done") return "results";
    if (snap.status === "error" || snap.status === "cancelled") return "error";
    return "processing";
  }, [submitError, jobId, snap]);

  const errorMessage = submitError ?? snap?.error ?? "Something went wrong.";

  const start = async (url: string, topic: string) => {
    setSubmitError(null);
    setSubmitting(true);
    try {
      const id = await createJob(url, topic, settings);
      setJobId(id);
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : "Could not start the job.");
    } finally {
      setSubmitting(false);
    }
  };

  const reset = () => {
    setJobId(null);
    setSubmitError(null);
  };

  return (
    <div className="min-h-full">
      <header className="mx-auto max-w-5xl px-4 sm:px-6 pt-6 flex items-center justify-between">
        <button onClick={reset} className="flex items-center gap-2 group">
          <span className="h-7 w-7 rounded-lg bg-gradient-to-br from-violet-500 to-cyan-400 shadow-lg" />
          <span className="font-semibold tracking-tight group-hover:text-white/90">Topic Clipper</span>
        </button>
        <button className="btn-ghost px-3 py-1.5 text-sm" onClick={() => setDrawerOpen(true)}>
          Settings
        </button>
      </header>

      <main className="mx-auto max-w-5xl px-4 sm:px-6 py-8 sm:py-12">
        {phase === "input" && <InputCard onSubmit={start} busy={submitting} />}

        {phase === "processing" && <ProcessingStepper snap={snap} />}

        {phase === "results" && snap && (
          <div className="animate-floatIn">
            <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-3 mb-6">
              <div>
                <h2 className="text-xl font-semibold">
                  {snap.clips.length} clip{snap.clips.length === 1 ? "" : "s"}
                </h2>
                {snap.notes ? <p className="text-sm text-white/50 mt-1">{snap.notes}</p> : null}
              </div>
              <div className="flex gap-2">
                <button className="btn-primary text-sm" onClick={reset}>
                  Clip another
                </button>
              </div>
            </div>
            <ResultsGrid clips={snap.clips} jobId={snap.job_id} />
          </div>
        )}

        {phase === "error" && (
          <div className="glass p-6 sm:p-8 w-full max-w-xl mx-auto text-center animate-floatIn">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-rose-500/15 text-rose-300 ring-1 ring-rose-400/30 text-xl">
              !
            </div>
            <h2 className="text-lg font-semibold">Couldn’t make clips</h2>
            <p className="mt-2 text-sm text-white/60">{errorMessage}</p>
            <button className="btn-primary mt-6" onClick={reset}>
              Try again
            </button>
          </div>
        )}
      </main>

      <SettingsDrawer
        open={drawerOpen}
        settings={settings}
        onChange={setSettings}
        onClose={() => setDrawerOpen(false)}
      />
    </div>
  );
}

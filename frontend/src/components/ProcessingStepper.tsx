import { JobSnapshot } from "../types";

const STAGES: [string, string][] = [
  ["transcribing", "Transcript"],
  ["selecting", "Selecting"],
  ["refining", "Refining"],
];

export function ProcessingStepper({ snap }: { snap: JobSnapshot | null }) {
  const stage = snap?.stage ?? "downloading";
  const pct = snap?.pct ?? 0;
  const message = snap?.message ?? "Starting…";
  const done = stage === "done";
  const currentIdx = done ? STAGES.length : STAGES.findIndex((s) => s[0] === stage);

  return (
    <div className="glass p-6 sm:p-8 w-full max-w-2xl mx-auto animate-floatIn">
      {snap?.title ? (
        <p className="text-sm text-white/50 mb-5 truncate" title={snap.title}>
          {snap.title}
        </p>
      ) : null}

      <ol className="flex flex-col md:flex-row md:items-center gap-3 md:gap-2">
        {STAGES.map(([key, label], i) => {
          const isDone = i < currentIdx;
          const isCurrent = i === currentIdx && !done;
          return (
            <li key={key} className="flex items-center gap-3 md:flex-col md:gap-2 md:flex-1">
              <div
                className={[
                  "flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-sm font-semibold transition",
                  isDone ? "bg-gradient-to-r from-violet-500 to-cyan-400 text-white" : "",
                  isCurrent ? "ring-2 ring-violet-400 bg-white/5 text-white animate-pulse" : "",
                  !isDone && !isCurrent ? "bg-white/5 text-white/40" : "",
                ].join(" ")}
              >
                {isDone ? "✓" : i + 1}
              </div>
              <span
                className={`text-sm md:text-center ${
                  isCurrent ? "text-white" : isDone ? "text-white/70" : "text-white/40"
                }`}
              >
                {label}
              </span>
            </li>
          );
        })}
      </ol>

      <div className="mt-6">
        <div className="h-2 w-full rounded-full bg-white/10 overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-violet-500 to-cyan-400 transition-[width] duration-500"
            style={{ width: `${Math.max(4, pct)}%` }}
          />
        </div>
        <div className="mt-3 flex items-center justify-between text-sm">
          <span className="text-white/70">{message}</span>
          <span className="text-white/40 tabular-nums">{Math.round(pct)}%</span>
        </div>
      </div>
    </div>
  );
}

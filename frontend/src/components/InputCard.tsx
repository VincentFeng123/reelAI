import { useState } from "react";

const YT = /(youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/shorts\/|youtube\.com\/embed\/|youtube\.com\/live\/|m\.youtube\.com\/watch\?v=)/i;

export function InputCard({
  onSubmit,
  busy,
}: {
  onSubmit: (url: string, topic: string) => void;
  busy?: boolean;
}) {
  const [url, setUrl] = useState("");
  const [topic, setTopic] = useState("");
  const [touched, setTouched] = useState(false);

  const urlValid = YT.test(url.trim());
  const ready = urlValid && topic.trim().length > 0;
  const showUrlError = touched && url.length > 0 && !urlValid;

  return (
    <div className="glass p-6 sm:p-8 w-full max-w-xl mx-auto animate-floatIn">
      <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight">
        Clip a video by{" "}
        <span className="bg-gradient-to-r from-violet-300 to-cyan-300 bg-clip-text text-transparent">
          topic
        </span>
      </h1>
      <p className="mt-2 text-sm text-white/50">
        Paste a YouTube link and a topic. The AI finds the most informative moments and
        cuts them into clean, separate clips.
      </p>

      <form
        className="mt-6 space-y-4"
        onSubmit={(e) => {
          e.preventDefault();
          if (ready && !busy) onSubmit(url.trim(), topic.trim());
        }}
      >
        <div>
          <label className="block text-xs font-medium text-white/60 mb-1.5">YouTube URL</label>
          <input
            className={`field ${showUrlError ? "ring-2 ring-rose-400/60 border-rose-400/40" : ""}`}
            placeholder="https://youtube.com/watch?v=…"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onBlur={() => setTouched(true)}
            inputMode="url"
            autoComplete="off"
          />
          {showUrlError && (
            <p className="mt-1.5 text-xs text-rose-300">That doesn’t look like a YouTube URL.</p>
          )}
        </div>

        <div>
          <label className="block text-xs font-medium text-white/60 mb-1.5">Topic</label>
          <input
            className="field"
            placeholder="e.g. momentum"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
          />
        </div>

        <button type="submit" className="btn-primary w-full" disabled={!ready || busy}>
          {busy ? "Starting…" : "Clip it"}
        </button>
      </form>
    </div>
  );
}

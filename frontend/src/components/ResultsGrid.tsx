import { Clip } from "../types";
import { ClipCard } from "./ClipCard";

export function ResultsGrid({ clips, jobId }: { clips: Clip[]; jobId: string }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
      {clips.map((c) => (
        <ClipCard key={c.n} clip={c} jobId={jobId} />
      ))}
    </div>
  );
}

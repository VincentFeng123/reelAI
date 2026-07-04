import { useEffect, useRef, useState } from "react";
import { JobSnapshot } from "../types";
import { getJob } from "../api";

/**
 * Subscribe to a job's SSE progress stream. Falls back to polling GET /jobs/{id}
 * if the EventSource transport repeatedly errors (e.g. flaky phone wifi).
 */
export function useJobStream(jobId: string | null): JobSnapshot | null {
  const [snap, setSnap] = useState<JobSnapshot | null>(null);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    if (!jobId) {
      setSnap(null);
      return;
    }
    const es = new EventSource(`/jobs/${jobId}/stream`);
    const apply = (e: MessageEvent) => {
      try {
        setSnap(JSON.parse(e.data));
      } catch {
        /* ignore malformed frame */
      }
    };
    const stopPoll = () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };

    es.addEventListener("progress", apply as EventListener);
    es.addEventListener("done", ((e: MessageEvent) => {
      apply(e);
      es.close();
    }) as EventListener);
    es.addEventListener("failed", ((e: MessageEvent) => {
      apply(e);
      es.close();
    }) as EventListener);

    let errs = 0;
    es.onerror = () => {
      errs += 1;
      if (errs >= 3 && !pollRef.current) {
        pollRef.current = window.setInterval(async () => {
          try {
            const s = await getJob(jobId);
            setSnap(s);
            if (s.status === "done" || s.status === "error" || s.status === "cancelled") {
              stopPoll();
            }
          } catch {
            /* keep polling */
          }
        }, 2000);
      }
    };

    return () => {
      es.close();
      stopPoll();
    };
  }, [jobId]);

  return snap;
}

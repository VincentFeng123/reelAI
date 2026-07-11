import { zipUrl } from "../api";

export function DownloadAll({ jobId }: { jobId: string }) {
  return (
    <a href={zipUrl(jobId)} download className="btn-ghost text-sm">
      ↓ Download all (zip)
    </a>
  );
}

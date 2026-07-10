import type { NextConfig } from "next";

const devBackendOrigin = (process.env.DEV_BACKEND_ORIGIN || "http://127.0.0.1:8000").replace(/\/$/, "");
const railwayBackendOrigin = (
  process.env.RAILWAY_BACKEND_ORIGIN
  || process.env.NEXT_PUBLIC_DEPLOYED_API_BASE
  || ""
).replace(/\/$/, "");

const nextConfig: NextConfig = {
  reactStrictMode: true,
  outputFileTracingRoot: process.cwd(),
  turbopack: {
    root: process.cwd(),
  },
  async rewrites() {
    const backendOrigin = process.env.NODE_ENV === "development"
      ? devBackendOrigin
      : railwayBackendOrigin;
    if (!backendOrigin) {
      return [];
    }
    return [
      {
        source: "/api/:path*",
        destination: `${backendOrigin}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;

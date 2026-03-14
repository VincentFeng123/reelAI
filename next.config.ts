import type { NextConfig } from "next";

const devBackendOrigin = (process.env.DEV_BACKEND_ORIGIN || "http://127.0.0.1:8000").replace(/\/$/, "");

const nextConfig: NextConfig = {
  reactStrictMode: true,
  outputFileTracingRoot: process.cwd(),
  turbopack: {
    root: process.cwd(),
  },
  async rewrites() {
    if (process.env.NODE_ENV !== "development") {
      return [];
    }
    return [
      {
        source: "/api/:path*",
        destination: `${devBackendOrigin}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;

import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#0D1B1E",
        sky: "#D9F2F7",
        pop: "#FF6B35",
        mint: "#2EC4B6",
        zinc: {
          925: "#111113",
        },
      },
      keyframes: {
        rise: {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        rise: "rise 420ms ease-out forwards",
      },
      boxShadow: {
        glow: "0 18px 46px rgba(46, 196, 182, 0.18)",
      },
    },
  },
  plugins: [],
};

export default config;

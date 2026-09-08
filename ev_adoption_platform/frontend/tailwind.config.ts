import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#090b0f",
        panel: "#111620",
        line: "#243041",
        signal: "#31d7a4",
        volt: "#f7c948",
        alert: "#ff5c7a",
        sky: "#52b7ff"
      }
    }
  },
  plugins: []
};

export default config;

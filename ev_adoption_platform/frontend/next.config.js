const configuredBasePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const basePath = configuredBasePath === "/" ? "" : configuredBasePath.replace(/\/$/, "");

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: "export",
  trailingSlash: true,
  basePath,
  assetPrefix: basePath || undefined,
  images: {
    unoptimized: true
  }
};

module.exports = nextConfig;

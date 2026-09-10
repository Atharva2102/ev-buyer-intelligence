import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { publicAsset } from "../lib/site";
import "./globals.css";

const geist = Geist({ subsets: ["latin"], variable: "--font-sans" });
const geistMono = Geist_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000"),
  title: "EV Buyer Intelligence",
  description: "An end-to-end data science portfolio project for EV adoption analytics, model scoring, scenario planning, and cloud monitoring.",
  openGraph: {
    title: "EV Buyer Intelligence",
    description: "From raw data and model validation to stakeholder-ready EV adoption decisions.",
    images: [publicAsset("/images/ev-hero.png")]
  }
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${geist.variable} ${geistMono.variable}`}>
      <body>{children}</body>
    </html>
  );
}

import type { Metadata } from "next";
import type { ReactNode } from "react";

import "@fortawesome/fontawesome-free/css/all.min.css";
import { GridGlowTracker } from "@/components/GridGlowTracker";
import "./globals.css";

export const metadata: Metadata = {
  title: "StudyReels",
  description: "Learn from your own material through short video reels.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        <GridGlowTracker />
        {children}
      </body>
    </html>
  );
}

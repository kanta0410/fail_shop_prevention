import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "出店場所提案アプリ",
  description: "AIが出店ポテンシャルを分析し、最適な立地を提案します。",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head />
      <body>{children}</body>
    </html>
  );
}

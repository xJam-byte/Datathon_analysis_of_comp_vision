import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CashGuard — AI Аудит кассовой дисциплины",
  description:
    "AI-сервис для выявления нарушений кассовой дисциплины по видеозаписям. ForteBank Datathon 26.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ru">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}

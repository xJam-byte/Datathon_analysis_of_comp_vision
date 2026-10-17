import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CashGuard — проверка кассовых видеозаписей",
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
          href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&subset=cyrillic&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}

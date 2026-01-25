import type { Metadata } from "next";
import { Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";
import { Toaster } from "sonner";

const jakarta = Plus_Jakarta_Sans({ subsets: ["latin"], variable: '--font-jakarta' });

export const metadata: Metadata = {
  title: "VirtualFit - AI Try-On Platform",
  description: "Experience the future of fashion with AI.",
  icons: {
    icon: '/favicon.ico',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={jakarta.className}>
        <Navbar />
        <main className="min-h-screen">
          {children}
        </main>
        <Toaster position="bottom-right" richColors closeButton />
      </body>
    </html>
  );
}

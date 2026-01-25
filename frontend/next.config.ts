import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '*.supabase.co',
      },
      {
        protocol: 'https',
        hostname: 'vlvuqcwjnjtqgqj.supabase.co', // Example Supabase Storage domain
      },
    ],
  },
  experimental: {
    // reactCompiler: true, // Commenting out if not using experimental React features yet or if it causes issues
  },
};

export default nextConfig;

"use client";
import { motion } from "framer-motion";
import Link from "next/link";

export default function Hero() {
  return (
    <div className="relative pt-32 pb-20 lg:pt-40 lg:pb-28 overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="text-center max-w-3xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <span className="bg-primary-50 text-primary-600 text-sm font-semibold px-4 py-1.5 rounded-full border border-primary-100 inline-block mb-6">
              AI-Powered Virtual Fitting Room
            </span>
            <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
              Try On
              <br />
              <span className="text-black">Anything</span>
            </h1>
            <p className="mt-4 text-xl text-gray-500 mb-10 max-w-2xl mx-auto leading-relaxed">
              Experience the future of fashion. Upload your photo and instantly try on
              any outfit in 3D using our advanced AI technology.
            </p>
            <div className="flex justify-center">
              <Link
                href="/onboard"
                className="btn-premium bg-black text-white px-8 py-4 rounded-xl font-semibold text-lg hover:bg-gray-800 transition-colors"
              >
                Get Started Free
              </Link>
            </div>
          </motion.div>
        </div>
      </div>
      
      {/* Abstract Background Elements */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-full overflow-hidden -z-10 pointer-events-none">
          <div className="absolute top-20 left-10 w-72 h-72 bg-primary-100/50 rounded-full blur-3xl mix-blend-multiply animate-pulse"></div>
          <div className="absolute top-40 right-10 w-96 h-96 bg-secondary-100/50 rounded-full blur-3xl mix-blend-multiply"></div>
      </div>
    </div>
  );
}

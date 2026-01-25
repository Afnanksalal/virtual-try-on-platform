"use client";
import { useState } from "react";
import TryOnWidget from "@/components/TryOnWidget";
import Recommendations from "@/components/Recommendations";
import { Zap, Download as DownloadIcon, ShoppingBag } from "lucide-react";
import Image from "next/image";
import { toast } from "sonner";
import ErrorBoundary from "@/components/ErrorBoundary";

import ProtectedRoute from "@/components/ProtectedRoute";

export default function StudioPage() {
  const [result, setResult] = useState<string | null>(null);
  
  const handleDownload = () => {
    if (!result) return;
    
    try {
      const link = document.createElement('a');
      link.href = result;
      link.download = `try-on-result-${Date.now()}.jpg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      toast.success("Image downloaded successfully!");
    } catch (error) {
      console.error("Download failed:", error);
      toast.error("Failed to download image");
    }
  };
  
  const handleBuyNow = () => {
    // In production, this would integrate with e-commerce platform
    // For now, redirect to shop page or show coming soon
    toast.info("E-commerce integration coming soon!");
  };
  
  return (
    <ProtectedRoute>
    <ErrorBoundary>
    <div className="min-h-screen pt-28 sm:pt-24 pb-12 px-4 sm:px-6 lg:px-8 bg-gray-50">
        <div className="max-w-7xl mx-auto">
            <div className="mb-6 sm:mb-8">
              <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-gray-900">Virtual Try-On Studio</h1>
              <p className="text-sm sm:text-base text-gray-600 mt-2">Upload a garment and see yourself wearing it</p>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8">
            
            {/* Left Col: Creation */}
            <div className="lg:col-span-3 space-y-6">
                <ErrorBoundary>
                  <TryOnWidget onResult={setResult} />
                </ErrorBoundary>
            </div>

            {/* Center Col: Visualization */}
            <div className="lg:col-span-6">
                <div className="bg-white rounded-3xl p-2 shadow-xl shadow-gray-200/50 min-h-[600px] flex items-center justify-center relative overflow-hidden">
                    {result ? (
                        <div className="relative w-full h-full">
                            <Image 
                              src={result} 
                              alt="Try-On Result" 
                              fill 
                              className="object-cover rounded-2xl" 
                              unoptimized // Since we might use external URLs or blobs
                            />
                            <div className="absolute top-4 right-4 flex gap-2 z-10">
                                <button 
                                  onClick={handleDownload}
                                  className="bg-white/90 backdrop-blur text-sm font-medium px-4 py-2 rounded-full shadow-sm hover:bg-white transition-colors flex items-center gap-2"
                                >
                                    <DownloadIcon className="h-4 w-4" />
                                    Download
                                </button>
                                <button 
                                  onClick={handleBuyNow}
                                  className="bg-black text-white text-sm font-medium px-4 py-2 rounded-full shadow-sm hover:bg-gray-800 transition-colors flex items-center gap-2"
                                >
                                    <ShoppingBag className="h-4 w-4" />
                                    Buy Now
                                </button>
                            </div>
                        </div>
                    ) : (
                         <div className="text-center text-gray-400">
                             <div className="h-24 w-24 bg-gray-100 rounded-full mx-auto mb-4 flex items-center justify-center animate-pulse">
                                 <Zap className="h-10 w-10 opacity-20" />
                             </div>
                             <p>Your creation will appear here</p>
                         </div>
                    )}
                </div>
                
                {result && (
                     <div className="mt-8">
                         <h3 className="text-lg font-bold mb-4 ml-2">3D View</h3>
                         {/* In production, we fetch the generated 3D model URL from the backend result metadata */}
                         {/* <ModelViewer modelUrl={result.model_3d_url} /> */}
                         <p className="text-gray-500 text-sm italic ml-2">3D reconstruction pending (Requires configured PIFuHD backend).</p>
                     </div>
                )}
            </div>

            {/* Right Col: AI Insights */}
            <div className="lg:col-span-3">
                <ErrorBoundary>
                  <Recommendations />
                </ErrorBoundary>
            </div>
          </div>
        </div>
    </div>
    </ErrorBoundary>
    </ProtectedRoute>
  );
}

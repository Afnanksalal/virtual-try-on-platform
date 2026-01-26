"use client";

import Image from "next/image";
import { Download, Sparkles } from "lucide-react";
import { TryOnResult } from "@/lib/types";
import { handleError, showSuccess } from "@/lib/errorHandling";

interface ResultsTabProps {
  results: TryOnResult[];
  isLoading?: boolean;
}

export default function ResultsTab({ results, isLoading = false }: ResultsTabProps) {
  const handleDownload = (result: TryOnResult) => {
    try {
      const link = document.createElement('a');
      link.href = result.resultUrl;
      link.download = `try-on-result-${result.id}-${Date.now()}.jpg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      showSuccess('Image downloaded successfully');
    } catch (error) {
      handleError(error, 'download result image', { showToast: true });
    }
  };

  if (isLoading) {
    return (
      <div 
        role="tabpanel" 
        id="results-panel" 
        aria-labelledby="results-tab"
        className="bg-white rounded-b-xl p-4 sm:p-6 md:p-8"
      >
        <div className="flex items-center justify-center h-64 sm:h-80 md:h-96">
          <div className="animate-pulse text-gray-400">Loading results...</div>
        </div>
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div 
        role="tabpanel" 
        id="results-panel" 
        aria-labelledby="results-tab"
        className="bg-white rounded-b-xl p-4 sm:p-6 md:p-8"
      >
        <div className="flex flex-col items-center justify-center h-64 sm:h-80 md:h-96 text-center px-4">
          <div className="h-20 w-20 sm:h-24 sm:w-24 bg-gray-100 rounded-full flex items-center justify-center mb-4">
            <Sparkles className="h-10 w-10 sm:h-12 sm:w-12 text-gray-400" />
          </div>
          <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-2">
            No Try-On Results Yet
          </h3>
          <p className="text-sm sm:text-base text-gray-600 max-w-md">
            Select a garment and click "Generate" to create your first virtual try-on result.
          </p>
        </div>
      </div>
    );
  }

  // Sort results by creation date (most recent first)
  const sortedResults = [...results].sort((a, b) => 
    new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
  );

  return (
    <div 
      role="tabpanel" 
      id="results-panel" 
      aria-labelledby="results-tab"
      className="bg-white rounded-b-xl p-4 sm:p-6 md:p-8"
    >
      <div className="max-w-6xl mx-auto">
        <div className="mb-4 sm:mb-6">
          <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-2">
            Your Try-On Results
          </h2>
          <p className="text-sm sm:text-base text-gray-600">
            {results.length} {results.length === 1 ? 'result' : 'results'} generated
          </p>
        </div>

        <div className="space-y-6 sm:space-y-8">
          {sortedResults.map((result) => (
            <div 
              key={result.id} 
              className="bg-gray-50 rounded-xl p-4 sm:p-6"
            >
              <div className="flex flex-col md:flex-row gap-4 sm:gap-6">
                {/* Original Garment */}
                <div className="flex-1">
                  <h3 className="text-xs sm:text-sm font-medium text-gray-700 mb-2 sm:mb-3">
                    Original Garment
                  </h3>
                  <div className="relative aspect-square bg-white rounded-lg overflow-hidden shadow-sm">
                    <Image
                      src={result.garmentUrl}
                      alt={`Original garment used for try-on result created on ${new Date(result.createdAt).toLocaleDateString()}`}
                      fill
                      className="object-cover"
                      loading="lazy"
                      unoptimized
                    />
                  </div>
                </div>

                {/* Try-On Result */}
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-2 sm:mb-3">
                    <h3 className="text-xs sm:text-sm font-medium text-gray-700">
                      Try-On Result
                    </h3>
                    <button
                      onClick={() => handleDownload(result)}
                      className="flex items-center gap-1 sm:gap-2 text-xs sm:text-sm font-medium text-gray-900 hover:text-black transition-colors min-h-[44px] px-2 sm:px-0"
                      aria-label={`Download try-on result from ${new Date(result.createdAt).toLocaleDateString()}`}
                    >
                      <Download className="h-4 w-4" aria-hidden="true" />
                      <span className="hidden sm:inline">Download</span>
                    </button>
                  </div>
                  <div className="relative aspect-square bg-white rounded-lg overflow-hidden shadow-sm">
                    <Image
                      src={result.resultUrl}
                      alt={`Virtual try-on result showing you wearing the garment, created on ${new Date(result.createdAt).toLocaleDateString()}`}
                      fill
                      className="object-cover"
                      loading="lazy"
                      unoptimized
                    />
                  </div>
                </div>
              </div>

              {/* Metadata */}
              <div className="mt-3 sm:mt-4 flex flex-col sm:flex-row flex-wrap gap-2 sm:gap-4 text-xs sm:text-sm text-gray-600">
                <div>
                  <span className="font-medium text-gray-900">Created:</span>{' '}
                  <span className="hidden sm:inline">{new Date(result.createdAt).toLocaleString()}</span>
                  <span className="sm:hidden">{new Date(result.createdAt).toLocaleDateString()}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-900">Status:</span>{' '}
                  <span className={`
                    capitalize
                    ${result.status === 'completed' ? 'text-green-600' : ''}
                    ${result.status === 'processing' ? 'text-yellow-600' : ''}
                    ${result.status === 'failed' ? 'text-red-600' : ''}
                  `}>
                    {result.status}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

"use client";

import Image from "next/image";
import { RefreshCw, Sparkles, ExternalLink } from "lucide-react";
import { Recommendation } from "@/lib/types";
import { handleError, showSuccess } from "@/lib/errorHandling";

interface RecommendationsTabProps {
  recommendations: Recommendation[];
  onRefresh: () => Promise<void>;
  onUseForTryOn: (recommendation: Recommendation) => Promise<void>;
  isLoading?: boolean;
}

export default function RecommendationsTab({
  recommendations,
  onRefresh,
  onUseForTryOn,
  isLoading = false
}: RecommendationsTabProps) {
  const handleRefresh = async () => {
    try {
      await onRefresh();
      showSuccess('Recommendations refreshed');
    } catch (error) {
      handleError(error, 'refresh recommendations', { showToast: true });
    }
  };

  const handleUseForTryOn = async (recommendation: Recommendation) => {
    try {
      await onUseForTryOn(recommendation);
      showSuccess('Garment added to your collection');
    } catch (error) {
      handleError(error, 'add recommendation to garments', { showToast: true });
    }
  };

  if (isLoading) {
    return (
      <div 
        role="tabpanel" 
        id="recommendations-panel" 
        aria-labelledby="recommendations-tab"
        className="bg-white rounded-b-xl p-4 sm:p-6 md:p-8"
      >
        <div className="flex items-center justify-center h-64 sm:h-80 md:h-96">
          <div className="flex flex-col items-center gap-3">
            <div className="animate-spin">
              <Sparkles className="h-8 w-8 text-gray-400" />
            </div>
            <p className="text-sm sm:text-base text-gray-400">Loading recommendations...</p>
          </div>
        </div>
      </div>
    );
  }

  if (recommendations.length === 0) {
    return (
      <div 
        role="tabpanel" 
        id="recommendations-panel" 
        aria-labelledby="recommendations-tab"
        className="bg-white rounded-b-xl p-4 sm:p-6 md:p-8"
      >
        <div className="flex flex-col items-center justify-center h-64 sm:h-80 md:h-96 text-center px-4">
          <div className="h-20 w-20 sm:h-24 sm:w-24 bg-gray-100 rounded-full flex items-center justify-center mb-4">
            <Sparkles className="h-10 w-10 sm:h-12 sm:w-12 text-gray-400" />
          </div>
          <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-2">
            No Recommendations Yet
          </h3>
          <p className="text-sm sm:text-base text-gray-600 max-w-md">
            Click the "Recommend" button to get AI-powered outfit suggestions.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div 
      role="tabpanel" 
      id="recommendations-panel" 
      aria-labelledby="recommendations-tab"
      className="bg-white rounded-b-xl p-4 sm:p-6 md:p-8"
    >
      <div className="max-w-6xl mx-auto">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 sm:mb-6 gap-3">
          <div>
            <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-1 sm:mb-2">
              AI Recommendations
            </h2>
            <p className="text-sm sm:text-base text-gray-600">
              {recommendations.length} {recommendations.length === 1 ? 'item' : 'items'} suggested for you
            </p>
          </div>
          <button
            onClick={handleRefresh}
            disabled={isLoading}
            className="flex items-center gap-2 px-4 py-2 bg-gray-900 text-white rounded-full font-medium hover:bg-gray-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed min-h-[44px] text-sm sm:text-base w-full sm:w-auto justify-center"
            aria-label="Refresh recommendations to get new suggestions"
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} aria-hidden="true" />
            Refresh
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
          {recommendations.map((recommendation) => (
            <div 
              key={recommendation.id} 
              className="bg-white border border-gray-200 rounded-xl overflow-hidden hover:shadow-lg transition-shadow"
            >
              {/* Product Image */}
              <div className="relative aspect-square bg-gray-100">
                <Image
                  src={recommendation.image_url}
                  alt={`${recommendation.name} in ${recommendation.category}, priced at ${recommendation.currency} ${recommendation.price.toFixed(2)}`}
                  fill
                  className="object-cover"
                  unoptimized
                />
              </div>

              {/* Product Info */}
              <div className="p-3 sm:p-4">
                <h3 className="font-semibold text-sm sm:text-base text-gray-900 mb-1 line-clamp-2">
                  {recommendation.name}
                </h3>
                <p className="text-xs sm:text-sm text-gray-600 mb-2">
                  {recommendation.category}
                </p>
                
                <div className="flex items-center justify-between mb-3">
                  <div className="text-base sm:text-lg font-bold text-gray-900">
                    {recommendation.currency} ${recommendation.price.toFixed(2)}
                  </div>
                  {recommendation.condition && (
                    <div className="text-xs text-gray-500">
                      {recommendation.condition}
                    </div>
                  )}
                </div>

                {/* Actions */}
                <div className="flex gap-2">
                  <button
                    onClick={() => handleUseForTryOn(recommendation)}
                    className="flex-1 bg-black text-white px-3 sm:px-4 py-2 rounded-lg text-xs sm:text-sm font-medium hover:bg-gray-800 transition-colors min-h-[44px]"
                    aria-label={`Add ${recommendation.name} to your garments for virtual try-on`}
                  >
                    Use for Try-On
                  </button>
                  <a
                    href={recommendation.ebay_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-center px-3 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors min-h-[44px] min-w-[44px]"
                    aria-label={`View ${recommendation.name} on eBay (opens in new tab)`}
                  >
                    <ExternalLink className="h-4 w-4 text-gray-600" aria-hidden="true" />
                  </a>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

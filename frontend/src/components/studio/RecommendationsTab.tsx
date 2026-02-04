"use client";

import Image from "next/image";
import { RefreshCw, Sparkles, ExternalLink, Package, Truck, Star } from "lucide-react";
import { Recommendation } from "@/lib/types";
import { handleError, showSuccess } from "@/lib/errorHandling";
import { useState } from "react";

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
  const [failedImages, setFailedImages] = useState<Set<string>>(new Set());

  const handleRefresh = async () => {
    try {
      setFailedImages(new Set()); // Reset failed images on refresh
      await onRefresh();
      showSuccess('Recommendations refreshed with your color palette');
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

  const handleImageError = (id: string) => {
    setFailedImages(prev => new Set([...prev, id]));
  };

  const getPlaceholderImage = (name: string) => {
    return `https://via.placeholder.com/400x500/f3f4f6/374151?text=${encodeURIComponent(name.slice(0, 20))}`;
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
            <p className="text-sm sm:text-base text-gray-500">Analyzing your skin tone and style...</p>
            <p className="text-xs text-gray-400">Finding color-coordinated recommendations</p>
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
          <div className="h-20 w-20 sm:h-24 sm:w-24 bg-gradient-to-br from-purple-100 to-pink-100 rounded-full flex items-center justify-center mb-4">
            <Sparkles className="h-10 w-10 sm:h-12 sm:w-12 text-purple-500" />
          </div>
          <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-2">
            Personalized Style Awaits
          </h3>
          <p className="text-sm sm:text-base text-gray-600 max-w-md">
            Click "Get Recommendations" to receive AI-powered outfit suggestions matched to your skin tone and style.
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
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 sm:mb-6 gap-3">
          <div>
            <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-1 sm:mb-2 flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-purple-500" />
              Color-Matched Recommendations
            </h2>
            <p className="text-sm sm:text-base text-gray-600">
              {recommendations.length} {recommendations.length === 1 ? 'item' : 'items'} curated for your skin tone
            </p>
          </div>
          <button
            onClick={handleRefresh}
            disabled={isLoading}
            className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-full font-medium hover:from-purple-700 hover:to-pink-700 transition-all disabled:opacity-50 disabled:cursor-not-allowed min-h-[44px] text-sm sm:text-base w-full sm:w-auto justify-center shadow-md hover:shadow-lg"
            aria-label="Refresh recommendations to get new suggestions"
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} aria-hidden="true" />
            New Recommendations
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 sm:gap-5">
          {recommendations.map((recommendation) => (
            <div 
              key={recommendation.id} 
              className="bg-white border border-gray-200 rounded-xl overflow-hidden hover:shadow-xl transition-all duration-300 group"
            >
              {/* Product Image */}
              <div className="relative aspect-[4/5] bg-gray-100 overflow-hidden">
                <Image
                  src={failedImages.has(recommendation.id) 
                    ? getPlaceholderImage(recommendation.name)
                    : recommendation.image_url
                  }
                  alt={`${recommendation.name} - ${recommendation.category}`}
                  fill
                  className="object-cover transition-transform duration-300 group-hover:scale-105"
                  unoptimized
                  onError={() => handleImageError(recommendation.id)}
                />
                {/* Relevance Badge */}
                {recommendation.relevance_score && recommendation.relevance_score > 70 && (
                  <div className="absolute top-2 left-2 bg-green-500 text-white text-xs px-2 py-1 rounded-full flex items-center gap-1">
                    <Star className="h-3 w-3" />
                    Top Match
                  </div>
                )}
                {/* Category Badge */}
                <div className="absolute bottom-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded-full backdrop-blur-sm">
                  {recommendation.category}
                </div>
              </div>

              {/* Product Info */}
              <div className="p-3 sm:p-4">
                <h3 className="font-semibold text-sm sm:text-base text-gray-900 mb-1 line-clamp-2 min-h-[2.5rem]">
                  {recommendation.name}
                </h3>
                
                <div className="flex items-center justify-between mb-3">
                  <div className="text-lg sm:text-xl font-bold text-gray-900">
                    ${recommendation.price.toFixed(2)}
                    <span className="text-xs text-gray-500 font-normal ml-1">{recommendation.currency}</span>
                  </div>
                </div>
                
                {/* Condition & Shipping */}
                <div className="flex items-center gap-3 mb-3 text-xs text-gray-500">
                  {recommendation.condition && recommendation.condition !== 'Unknown' && (
                    <div className="flex items-center gap-1">
                      <Package className="h-3 w-3" />
                      {recommendation.condition}
                    </div>
                  )}
                  {recommendation.shipping !== undefined && (
                    <div className="flex items-center gap-1">
                      <Truck className="h-3 w-3" />
                      {recommendation.shipping === 0 ? (
                        <span className="text-green-600 font-medium">Free Shipping</span>
                      ) : (
                        <span>+${recommendation.shipping.toFixed(2)}</span>
                      )}
                    </div>
                  )}
                </div>

                {/* Actions */}
                <div className="flex gap-2">
                  <button
                    onClick={() => handleUseForTryOn(recommendation)}
                    className="flex-1 bg-black text-white px-3 sm:px-4 py-2.5 rounded-lg text-xs sm:text-sm font-medium hover:bg-gray-800 transition-colors min-h-[44px]"
                    aria-label={`Add ${recommendation.name} to your garments for virtual try-on`}
                  >
                    Try On
                  </button>
                  <a
                    href={recommendation.ebay_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-center px-3 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors min-h-[44px] min-w-[44px] text-xs sm:text-sm font-medium gap-1"
                    aria-label={`Buy ${recommendation.name} on eBay (opens in new tab)`}
                  >
                    <ExternalLink className="h-4 w-4" aria-hidden="true" />
                    <span className="hidden sm:inline">Buy</span>
                  </a>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Info Footer */}
        <div className="mt-6 p-4 bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl">
          <p className="text-sm text-gray-600 text-center">
            <Sparkles className="h-4 w-4 inline-block mr-1 text-purple-500" />
            Recommendations are based on color theory analysis of your skin tone for optimal style matching.
          </p>
        </div>
      </div>
    </div>
  );
}

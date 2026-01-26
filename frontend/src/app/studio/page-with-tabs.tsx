"use client";

import { useState } from "react";
import { toast } from "sonner";
import ErrorBoundary from "@/components/ErrorBoundary";
import ProtectedRoute from "@/components/ProtectedRoute";
import {
  TabNavigation,
  PersonalImageTab,
  GarmentTab,
  ResultsTab,
  RecommendationsTab,
  TabType
} from "@/components/studio";
import { Garment, TryOnResult, Recommendation } from "@/lib/types";

/**
 * Example Studio Page with Tab Interface
 * 
 * This is a demonstration of how to integrate the tab components.
 * The actual implementation in task 12 will include full state management,
 * API integration, and proper data flow.
 */
export default function StudioPageWithTabs() {
  // Tab state
  const [activeTab, setActiveTab] = useState<TabType>('personal');
  const [showRecommendations, setShowRecommendations] = useState(false);

  // Data state (mock data for demonstration)
  const [personalImage] = useState<{
    url: string;
    type: 'head-only' | 'full-body';
    uploadedAt: Date;
  } | null>(null);
  
  const [garments] = useState<Garment[]>([]);
  const [selectedGarment, setSelectedGarment] = useState<Garment | null>(null);
  const [tryOnResults] = useState<TryOnResult[]>([]);
  const [recommendations] = useState<Recommendation[]>([]);

  // Loading states
  const [isGenerating, setIsGenerating] = useState(false);
  const [isLoadingRecommendations, setIsLoadingRecommendations] = useState(false);

  // Handlers (to be implemented in task 12)
  const handleTabChange = (tab: TabType) => {
    setActiveTab(tab);
  };

  const handlePersonalImageUpload = async (file: File) => {
    // TODO: Implement in task 12
    console.log('Upload personal image:', file);
  };

  const handleGarmentUpload = async (file: File) => {
    // TODO: Implement in task 12
    console.log('Upload garment:', file);
  };

  const handleGarmentSelect = (garment: Garment) => {
    setSelectedGarment(garment);
  };

  const handleGenerateTryOn = async () => {
    if (!selectedGarment) {
      toast.error('Please select a garment first');
      return;
    }

    try {
      setIsGenerating(true);
      // TODO: Implement API call in task 12
      toast.success('Try-on generated successfully');
      setActiveTab('results');
    } catch (error) {
      console.error('Generation failed:', error);
      toast.error('Failed to generate try-on');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRequestRecommendations = async () => {
    try {
      setIsLoadingRecommendations(true);
      // TODO: Implement API call in task 12
      setShowRecommendations(true);
      setActiveTab('recommendations');
      toast.success('Recommendations loaded');
    } catch (error) {
      console.error('Failed to load recommendations:', error);
      toast.error('Failed to load recommendations');
    } finally {
      setIsLoadingRecommendations(false);
    }
  };

  const handleRefreshRecommendations = async () => {
    // TODO: Implement in task 12
    console.log('Refresh recommendations');
  };

  const handleUseRecommendationForTryOn = async (recommendation: Recommendation) => {
    // TODO: Implement in task 12
    console.log('Use recommendation:', recommendation);
  };

  return (
    <ProtectedRoute>
      <ErrorBoundary>
        <div className="min-h-screen pt-28 sm:pt-24 pb-12 px-4 sm:px-6 lg:px-8 bg-gray-50">
          <div className="max-w-7xl mx-auto">
            <div className="mb-6 sm:mb-8">
              <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-gray-900">
                Virtual Try-On Studio
              </h1>
              <p className="text-sm sm:text-base text-gray-600 mt-2">
                Professional workspace for virtual try-on
              </p>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-4 mb-6">
              <button
                onClick={handleGenerateTryOn}
                disabled={!selectedGarment || isGenerating}
                className="px-6 py-3 bg-black text-white rounded-full font-medium hover:bg-gray-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isGenerating ? 'Generating...' : 'Generate Try-On'}
              </button>
              <button
                onClick={handleRequestRecommendations}
                disabled={isLoadingRecommendations}
                className="px-6 py-3 bg-white text-gray-900 border border-gray-300 rounded-full font-medium hover:bg-gray-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoadingRecommendations ? 'Loading...' : 'Get Recommendations'}
              </button>
            </div>

            {/* Tab Interface */}
            <div className="bg-white rounded-xl shadow-sm">
              <TabNavigation
                activeTab={activeTab}
                onTabChange={handleTabChange}
                showRecommendations={showRecommendations}
              />

              {activeTab === 'personal' && (
                <PersonalImageTab
                  personalImage={personalImage}
                  onImageUpload={handlePersonalImageUpload}
                />
              )}

              {activeTab === 'garment' && (
                <GarmentTab
                  garments={garments}
                  selectedGarment={selectedGarment}
                  onGarmentSelect={handleGarmentSelect}
                  onGarmentUpload={handleGarmentUpload}
                />
              )}

              {activeTab === 'results' && (
                <ResultsTab results={tryOnResults} />
              )}

              {activeTab === 'recommendations' && (
                <RecommendationsTab
                  recommendations={recommendations}
                  onRefresh={handleRefreshRecommendations}
                  onUseForTryOn={handleUseRecommendationForTryOn}
                  isLoading={isLoadingRecommendations}
                />
              )}
            </div>
          </div>
        </div>
      </ErrorBoundary>
    </ProtectedRoute>
  );
}

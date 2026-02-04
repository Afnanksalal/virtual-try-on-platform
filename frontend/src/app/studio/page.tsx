"use client";

import { useState, useEffect } from "react";
import { endpoints } from "@/lib/api";
import { handleError, showSuccess, showInfo } from "@/lib/errorHandling";
import ErrorBoundary from "@/components/ErrorBoundary";
import ProtectedRoute from "@/components/ProtectedRoute";
import { StudioProvider, useStudio } from "@/contexts/StudioContext";
import {
  TabNavigation,
  PersonalImageTab,
  GarmentTab,
  ResultsTab,
  RecommendationsTab,
  type TabType,
} from "@/components/studio";
import type { Garment, Recommendation } from "@/lib/types";
import { Sparkles, Zap } from "lucide-react";

function StudioContent() {
  const {
    personalImage,
    userProfile,
    garments,
    tryOnResults,
    recommendations,
    selectedGarment,
    isGenerating,
    isLoadingRecommendations,
    hasRequestedRecommendations,
    isLoadingGarments,
    isLoadingPersonalImage,
    userId,
    addGarment,
    removeGarment,
    selectGarment,
    addTryOnResult,
    setIsGenerating,
    setRecommendations,
    setIsLoadingRecommendations,
    setHasRequestedRecommendations,
    setPersonalImage,
  } = useStudio();

  const [activeTab, setActiveTab] = useState<TabType>('personal');

  // Check for pre-selected garment from Wardrobe
  useEffect(() => {
    const preSelectedGarment = sessionStorage.getItem('selectedGarment');
    if (preSelectedGarment && garments.length > 0) {
      try {
        const garment = JSON.parse(preSelectedGarment);
        const matchingGarment = garments.find(g => g.id === garment.id);
        if (matchingGarment) {
          selectGarment(matchingGarment);
          setActiveTab('garment');
          showSuccess('Garment loaded from Wardrobe');
        }
        sessionStorage.removeItem('selectedGarment');
      } catch (error) {
        handleError(error, 'load pre-selected garment', { showToast: false });
      }
    }
  }, [garments, selectGarment]);

  // Tab change handler
  const handleTabChange = (tab: TabType) => {
    setActiveTab(tab);
  };

  // Garment selection handler
  const handleGarmentSelect = (garment: Garment) => {
    selectGarment(garment);
  };

  // Garment upload handler with optimistic update
  const handleGarmentUpload = async (file: File) => {
    if (!userId) {
      handleError(
        new Error('Not authenticated'),
        'upload garment',
        { showToast: true, userId: userId || undefined }
      );
      return;
    }

    // Optimistic update: create temporary garment
    const tempId = `temp-${Date.now()}`;
    const tempUrl = URL.createObjectURL(file);
    const tempGarment: Garment = {
      id: tempId,
      userId,
      url: tempUrl,
      thumbnailUrl: tempUrl,
      name: file.name,
      uploadedAt: new Date(),
      metadata: {
        width: 0,
        height: 0,
        size: file.size,
        format: file.type,
      },
    };

    // Add optimistically
    addGarment(tempGarment);
    showInfo('Uploading garment...');

    try {
      const newGarment = await endpoints.uploadGarment(file, userId);
      
      // Remove temp garment and add real one
      removeGarment(tempId);
      addGarment(newGarment);
      
      showSuccess('Garment uploaded successfully');
      
      // Clean up temp URL
      URL.revokeObjectURL(tempUrl);
    } catch (error) {
      handleError(error, 'upload garment', { showToast: true, userId });
      
      // Remove temp garment on error
      removeGarment(tempId);
      URL.revokeObjectURL(tempUrl);
      throw error;
    }
  };

  // Generate try-on handler
  const handleGenerateTryOn = async () => {
    if (!personalImage || !selectedGarment) {
      handleError(
        new Error('Missing required data'),
        'generate try-on',
        { showToast: true, userId: userId || undefined }
      );
      return;
    }

    try {
      setIsGenerating(true);
      showInfo('Generating try-on result...');

      const result = await endpoints.generateTryOn(
        personalImage.url,
        selectedGarment.url
      );

      addTryOnResult(result);
      setIsGenerating(false);
      setActiveTab('results'); // Auto-switch to results tab

      showSuccess('Try-on result generated successfully');
    } catch (error) {
      handleError(error, 'generate try-on', { showToast: true, userId: userId || undefined });
      setIsGenerating(false);
    }
  };

  // Request recommendations handler
  const handleRequestRecommendations = async () => {
    if (!personalImage) {
      handleError(
        new Error('No personal image'),
        'request recommendations',
        { showToast: true, userId: userId || undefined }
      );
      return;
    }

    try {
      setIsLoadingRecommendations(true);
      setHasRequestedRecommendations(true);
      showInfo('Fetching AI recommendations...');

      // Convert personal image URL to File
      const response = await fetch(personalImage.url);
      const blob = await response.blob();
      const personalImageFile = new File([blob], 'personal.jpg', { type: blob.type });

      // Pass user profile data for personalized color theory recommendations
      const recommendations = await endpoints.getRecommendations(
        personalImageFile,
        undefined, // wardrobeImages
        undefined, // generatedImages
        userProfile || undefined
      );

      setRecommendations(recommendations);
      setIsLoadingRecommendations(false);
      setActiveTab('recommendations'); // Auto-switch to recommendations tab

      showSuccess('Recommendations loaded successfully');
    } catch (error) {
      handleError(error, 'fetch recommendations', { showToast: true, userId: userId || undefined });
      setIsLoadingRecommendations(false);
    }
  };

  // Refresh recommendations handler
  const handleRefreshRecommendations = async () => {
    await handleRequestRecommendations();
  };

  // Use recommendation for try-on handler
  const handleUseRecommendationForTryOn = async (recommendation: Recommendation) => {
    if (!userId) {
      handleError(
        new Error('Not authenticated'),
        'use recommendation',
        { showToast: true }
      );
      return;
    }

    try {
      // Download recommendation image
      const response = await fetch(recommendation.image_url);
      const blob = await response.blob();
      const file = new File([blob], `${recommendation.name}.jpg`, { type: blob.type });

      // Upload as garment
      await handleGarmentUpload(file);

      // Switch to garment tab
      setActiveTab('garment');
    } catch (error) {
      handleError(error, 'use recommendation for try-on', { showToast: true, userId });
      throw error;
    }
  };

  // Personal image upload handler
  const handlePersonalImageUpload = async (file: File) => {
    if (!userId) {
      handleError(
        new Error('Not authenticated'),
        'upload personal image',
        { showToast: true }
      );
      return;
    }

    try {
      const result = await endpoints.updatePersonalImage(userId, file);
      setPersonalImage(result);
      showSuccess('Personal image updated successfully');
    } catch (error) {
      handleError(error, 'update personal image', { showToast: true, userId });
      throw error;
    }
  };

  // Check if Generate button should be enabled
  const canGenerate = selectedGarment !== null && personalImage !== null && !isGenerating;

  return (
    <ProtectedRoute>
      <ErrorBoundary>
        <div className="min-h-screen pt-20 sm:pt-24 pb-8 sm:pb-12 px-4 sm:px-6 lg:px-8 bg-gray-50">
          <div className="max-w-7xl mx-auto">
            {/* Header */}
            <div className="mb-4 sm:mb-6 md:mb-8">
              <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-gray-900">
                Virtual Try-On Studio
              </h1>
              <p className="text-sm sm:text-base text-gray-600 mt-1 sm:mt-2">
                Create professional virtual try-on results with AI
              </p>
            </div>

            {/* Action Buttons */}
            <div className="mb-4 sm:mb-6 flex flex-col sm:flex-row gap-3 sm:gap-4">
              <button
                onClick={handleGenerateTryOn}
                disabled={!canGenerate}
                className={`
                  flex items-center justify-center gap-2 px-6 py-3 rounded-full font-medium transition-all min-h-[44px] text-sm sm:text-base
                  ${canGenerate
                    ? 'bg-black text-white hover:bg-gray-800 shadow-lg hover:shadow-xl'
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  }
                `}
                aria-label={isGenerating ? 'Generating try-on result, please wait' : 'Generate virtual try-on result'}
                aria-busy={isGenerating}
              >
                <Zap className={`h-4 w-4 sm:h-5 sm:w-5 ${isGenerating ? 'animate-pulse' : ''}`} aria-hidden="true" />
                {isGenerating ? 'Generating...' : 'Generate Try-On'}
              </button>

              <button
                onClick={handleRequestRecommendations}
                disabled={isLoadingRecommendations || !personalImage}
                className={`
                  flex items-center justify-center gap-2 px-6 py-3 rounded-full font-medium transition-all min-h-[44px] text-sm sm:text-base
                  ${personalImage && !isLoadingRecommendations
                    ? 'bg-white text-gray-900 border-2 border-gray-900 hover:bg-gray-50'
                    : 'bg-gray-200 text-gray-500 border-2 border-gray-300 cursor-not-allowed'
                  }
                `}
                aria-label={isLoadingRecommendations ? 'Loading AI recommendations, please wait' : 'Get AI-powered outfit recommendations'}
                aria-busy={isLoadingRecommendations}
              >
                <Sparkles className={`h-4 w-4 sm:h-5 sm:w-5 ${isLoadingRecommendations ? 'animate-spin' : ''}`} aria-hidden="true" />
                {isLoadingRecommendations ? 'Loading...' : 'Get Recommendations'}
              </button>
            </div>

            {/* Screen reader announcements for loading states */}
            <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
              {isGenerating && 'Generating try-on result, please wait'}
              {isLoadingRecommendations && 'Loading recommendations, please wait'}
            </div>

            {/* Tab Interface */}
            <div className="bg-white rounded-xl shadow-lg overflow-hidden">
              <TabNavigation
                activeTab={activeTab}
                onTabChange={handleTabChange}
                showRecommendations={hasRequestedRecommendations}
              />

              {/* Tab Content */}
              <div className="min-h-[400px] sm:min-h-[500px] md:min-h-[600px]">
                {activeTab === 'personal' && (
                  <PersonalImageTab
                    personalImage={personalImage}
                    onImageUpload={handlePersonalImageUpload}
                    isLoading={isLoadingPersonalImage}
                  />
                )}

                {activeTab === 'garment' && (
                  <GarmentTab
                    garments={garments}
                    selectedGarment={selectedGarment}
                    onGarmentSelect={handleGarmentSelect}
                    onGarmentUpload={handleGarmentUpload}
                    isLoading={isLoadingGarments}
                  />
                )}

                {activeTab === 'results' && (
                  <ResultsTab
                    results={tryOnResults}
                    isLoading={false}
                  />
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
        </div>
      </ErrorBoundary>
    </ProtectedRoute>
  );
}

export default function StudioPage() {
  return (
    <StudioProvider>
      <StudioContent />
    </StudioProvider>
  );
}

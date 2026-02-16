"use client";

import Image from "next/image";
import { Download, Sparkles, Box, Loader2 } from "lucide-react";
import { TryOnResult, ThreeDModel } from "@/lib/types";
import { handleError, showSuccess, showInfo } from "@/lib/errorHandling";
import { useState } from "react";
import { endpoints } from "@/lib/api";
import ModelViewer from "@/components/ModelViewer";

interface ResultsTabProps {
  results: TryOnResult[];
  isLoading?: boolean;
}

export default function ResultsTab({ results, isLoading = false }: ResultsTabProps) {
  const [generating3D, setGenerating3D] = useState<string | null>(null);
  const [threeDModels, setThreeDModels] = useState<Map<string, ThreeDModel>>(new Map());
  const [viewing3D, setViewing3D] = useState<string | null>(null);

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

  const handleGenerate3D = async (result: TryOnResult) => {
    if (generating3D) return;

    try {
      setGenerating3D(result.id);
      showInfo('Generating 3D model... This may take up to 60 seconds');

      const response = await endpoints.generate3D(result.resultUrl, 'glb');
      
      // Prepend API base URL to the relative download_url
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const fullDownloadUrl = `${API_BASE_URL}${response.download_url}`;
      
      // Process intermediate images if available
      const intermediateImages: ThreeDModel['intermediateImages'] = {};
      if (response.metadata?.intermediate_images) {
        const intermediate = response.metadata.intermediate_images;
        if (intermediate.depth_map) {
          intermediateImages.depth_map = `${API_BASE_URL}${intermediate.depth_map.url}`;
        }
        if (intermediate.segmented_rgba) {
          intermediateImages.segmented_rgba = `${API_BASE_URL}${intermediate.segmented_rgba.url}`;
        }
        if (intermediate.mask_visualization) {
          intermediateImages.mask_visualization = `${API_BASE_URL}${intermediate.mask_visualization.url}`;
        }
      }
      
      console.log('3D Model Generation Response:', {
        download_url: response.download_url,
        download_token: response.download_token,
        fullDownloadUrl,
        API_BASE_URL,
        intermediateImages
      });
      
      const newModel: ThreeDModel = {
        url: fullDownloadUrl,
        format: response.format as 'glb' | 'obj' | 'ply',
        downloadToken: response.download_token, // Use the token directly from response
        expiresAt: new Date(response.expires_at),
        intermediateImages: Object.keys(intermediateImages).length > 0 ? intermediateImages : undefined
      };

      setThreeDModels(prev => new Map(prev).set(result.id, newModel));
      setViewing3D(result.id);
      showSuccess('3D model generated successfully!');
    } catch (error) {
      handleError(error, 'generate 3D model', { showToast: true });
    } finally {
      setGenerating3D(null);
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
          {sortedResults.map((result) => {
            const threeDModel = threeDModels.get(result.id);
            const isViewing3D = viewing3D === result.id;
            const isGenerating = generating3D === result.id;

            return (
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

                {/* 3D Generation Button */}
                <div className="mt-4 sm:mt-6 flex flex-col sm:flex-row gap-3">
                  <button
                    onClick={() => handleGenerate3D(result)}
                    disabled={isGenerating || !!threeDModel}
                    className={`
                      flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg font-medium transition-all text-sm
                      ${threeDModel 
                        ? 'bg-green-100 text-green-700 cursor-default' 
                        : isGenerating
                        ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
                        : 'bg-black text-white hover:bg-gray-800'
                      }
                    `}
                  >
                    {isGenerating ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Generating 3D Model...
                      </>
                    ) : threeDModel ? (
                      <>
                        <Box className="h-4 w-4" />
                        3D Model Ready
                      </>
                    ) : (
                      <>
                        <Box className="h-4 w-4" />
                        Generate 3D Model
                      </>
                    )}
                  </button>

                  {threeDModel && (
                    <button
                      onClick={() => setViewing3D(isViewing3D ? null : result.id)}
                      className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg font-medium transition-all text-sm bg-white border-2 border-gray-900 text-gray-900 hover:bg-gray-50"
                    >
                      {isViewing3D ? 'Hide 3D View' : 'View 3D Model'}
                    </button>
                  )}
                </div>

                {/* 3D Model Viewer */}
                {isViewing3D && threeDModel && (
                  <div className="mt-4 sm:mt-6 space-y-4">
                    <ModelViewer
                      modelUrl={threeDModel.url}
                      format={threeDModel.format}
                      downloadToken={threeDModel.downloadToken}
                    />
                    {threeDModel.expiresAt && (
                      <p className="mt-2 text-xs text-gray-500 text-center">
                        Download link expires: {threeDModel.expiresAt.toLocaleString()}
                      </p>
                    )}
                    
                    {/* Advanced Steps - Show intermediate processing images */}
                    {threeDModel.intermediateImages && (
                      <div className="mt-6 border-t pt-6">
                        <h4 className="text-sm font-semibold text-gray-900 mb-4">
                          Advanced Processing Steps
                        </h4>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          {threeDModel.intermediateImages.depth_map && (
                            <div className="space-y-2">
                              <h5 className="text-xs font-medium text-gray-700">
                                1. Depth Map (Depth Anything V2)
                              </h5>
                              <div className="relative aspect-square bg-gray-100 rounded-lg overflow-hidden">
                                <Image
                                  src={threeDModel.intermediateImages.depth_map}
                                  alt="Depth map showing distance from camera"
                                  fill
                                  className="object-cover"
                                  unoptimized
                                />
                              </div>
                              <p className="text-xs text-gray-500">
                                Warmer colors = closer to camera
                              </p>
                            </div>
                          )}
                          
                          {threeDModel.intermediateImages.mask_visualization && (
                            <div className="space-y-2">
                              <h5 className="text-xs font-medium text-gray-700">
                                2. Segmentation Mask (SAM2)
                              </h5>
                              <div className="relative aspect-square bg-gray-100 rounded-lg overflow-hidden">
                                <Image
                                  src={threeDModel.intermediateImages.mask_visualization}
                                  alt="Segmentation mask showing detected object"
                                  fill
                                  className="object-cover"
                                  unoptimized
                                />
                              </div>
                              <p className="text-xs text-gray-500">
                                Cyan overlay = detected object
                              </p>
                            </div>
                          )}
                          
                          {threeDModel.intermediateImages.segmented_rgba && (
                            <div className="space-y-2">
                              <h5 className="text-xs font-medium text-gray-700">
                                3. Segmented Image (RGBA)
                              </h5>
                              <div className="relative aspect-square bg-gray-100 rounded-lg overflow-hidden">
                                <Image
                                  src={threeDModel.intermediateImages.segmented_rgba}
                                  alt="Segmented image with transparent background"
                                  fill
                                  className="object-cover"
                                  unoptimized
                                />
                              </div>
                              <p className="text-xs text-gray-500">
                                Background removed, ready for 3D
                              </p>
                            </div>
                          )}
                        </div>
                        <p className="mt-4 text-xs text-gray-500 italic">
                          These intermediate steps show how the AI processes your image before generating the 3D model.
                        </p>
                      </div>
                    )}
                  </div>
                )}

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
            );
          })}
        </div>
      </div>
    </div>
  );
}

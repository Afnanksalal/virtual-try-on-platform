"use client";

import { useState, useRef } from "react";
import Image from "next/image";
import { Upload, Check, ChevronDown, ChevronUp, Zap, Paintbrush } from "lucide-react";
import { Garment, GarmentType, ModelType, TryOnOptions } from "@/lib/types";
import { validateGarmentImage } from "@/lib/validation";
import { handleError, showSuccess, showInfo } from "@/lib/errorHandling";

interface GarmentTabProps {
  garments: Garment[];
  selectedGarment: Garment | null;
  onGarmentSelect: (garment: Garment) => void;
  onGarmentUpload: (file: File) => Promise<void>;
  isLoading?: boolean;
  tryOnOptions: TryOnOptions;
  onOptionsChange: <K extends keyof TryOnOptions>(key: K, value: TryOnOptions[K]) => void;
}

export default function GarmentTab({
  garments,
  selectedGarment,
  onGarmentSelect,
  onGarmentUpload,
  isLoading = false,
  tryOnOptions,
  onOptionsChange
}: GarmentTabProps) {
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (file: File) => {
    // Validate file
    const validationResult = await validateGarmentImage(file);
    if (!validationResult.isValid) {
      handleError(
        new Error(validationResult.error),
        'validate garment image',
        { showToast: true }
      );
      return;
    }

    try {
      setUploading(true);
      showInfo('Uploading garment...');
      await onGarmentUpload(file);
      showSuccess('Garment uploaded successfully');
    } catch (error) {
      handleError(error, 'upload garment', { showToast: true });
    } finally {
      setUploading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  return (
    <div 
      role="tabpanel" 
      id="garment-panel" 
      aria-labelledby="garment-tab"
      className="bg-white rounded-b-xl p-4 sm:p-6 md:p-8"
    >
      <div className="max-w-4xl mx-auto">
        {/* Upload Section */}
        <div className="mb-6 sm:mb-8">
          <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-3 sm:mb-4">
            Upload Garment
          </h2>
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            className={`
              border-2 border-dashed rounded-xl p-6 sm:p-8 text-center transition-colors
              ${dragActive 
                ? 'border-black bg-gray-50' 
                : 'border-gray-300 hover:border-gray-400'
              }
              ${uploading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
            onClick={() => !uploading && fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/jpg"
              onChange={handleFileChange}
              className="hidden"
              disabled={uploading}
              aria-label="Upload garment image"
            />
            <Upload className="h-10 w-10 sm:h-12 sm:w-12 text-gray-400 mx-auto mb-3 sm:mb-4" aria-hidden="true" />
            <p className="text-sm sm:text-base text-gray-900 font-medium mb-1">
              {uploading ? 'Uploading...' : 'Drop your garment image here'}
            </p>
            <p className="text-xs sm:text-sm text-gray-500">
              or click to browse (JPG, PNG • Max 10MB)
            </p>
          </div>
        </div>

        {/* Selected Garment Preview */}
        {selectedGarment && (
          <div className="mb-6 sm:mb-8 bg-gray-50 rounded-xl p-4 sm:p-6">
            <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-3 sm:mb-4">
              Selected Garment
            </h3>
            <div className="relative aspect-square max-w-xs mx-auto bg-white rounded-lg overflow-hidden shadow-sm">
              <Image
                src={selectedGarment.url}
                alt={`Selected garment: ${selectedGarment.name}`}
                fill
                className="object-cover"
                unoptimized
              />
            </div>
          </div>
        )}

        {/* Try-On Options */}
        <div className="mb-6 sm:mb-8">
          <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-3 sm:mb-4">
            Try-On Settings
          </h2>
          
          {/* Garment Type Selector */}
          <div className="mb-4">
            <label className="text-sm font-medium text-gray-700 mb-2 block">Garment Type</label>
            <div className="grid grid-cols-3 gap-2">
              {[
                { value: "upper_body", label: "Top" },
                { value: "lower_body", label: "Bottom" },
                { value: "dresses", label: "Dress/Full" },
              ].map((option) => (
                <button
                  key={option.value}
                  onClick={() => onOptionsChange("garment_type", option.value as GarmentType)}
                  className={`py-2 px-3 rounded-lg text-sm font-medium transition-all ${
                    tryOnOptions.garment_type === option.value
                      ? "bg-black text-white"
                      : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          {/* Advanced Options Toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="w-full flex items-center justify-between py-2 px-3 mb-4 text-sm text-gray-600 hover:text-gray-800 transition-colors border border-gray-200 rounded-lg"
          >
            <span className="font-medium">Advanced Options</span>
            {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>

          {/* Advanced Options Panel */}
          {showAdvanced && (
            <div className="p-4 bg-gray-50 rounded-xl space-y-4">
              {/* Model Type */}
              <div>
                <label className="text-sm font-medium text-gray-700 mb-2 block">Model</label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => onOptionsChange("model_type", "viton_hd")}
                    className={`py-2 px-3 rounded-lg text-sm font-medium transition-all ${
                      tryOnOptions.model_type === "viton_hd"
                        ? "bg-black text-white"
                        : "bg-white text-gray-600 hover:bg-gray-100 border border-gray-200"
                    }`}
                  >
                    VITON-HD
                  </button>
                  <button
                    onClick={() => onOptionsChange("model_type", "dress_code")}
                    className={`py-2 px-3 rounded-lg text-sm font-medium transition-all ${
                      tryOnOptions.model_type === "dress_code"
                        ? "bg-black text-white"
                        : "bg-white text-gray-600 hover:bg-gray-100 border border-gray-200"
                    }`}
                  >
                    DressCode
                  </button>
                </div>
              </div>

              {/* Inference Steps */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm font-medium text-gray-700">Steps</label>
                  <span className="text-sm text-gray-500">{tryOnOptions.num_inference_steps || 30}</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="50"
                  value={tryOnOptions.num_inference_steps || 30}
                  onChange={(e) => onOptionsChange("num_inference_steps", Number(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-black"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>Fast (10)</span>
                  <span>Quality (50)</span>
                </div>
              </div>

              {/* Guidance Scale */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm font-medium text-gray-700">Guidance</label>
                  <span className="text-sm text-gray-500">{(tryOnOptions.guidance_scale || 2.5).toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="5"
                  step="0.1"
                  value={tryOnOptions.guidance_scale || 2.5}
                  onChange={(e) => onOptionsChange("guidance_scale", Number(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-black"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>Creative (1.0)</span>
                  <span>Precise (5.0)</span>
                </div>
              </div>

              {/* Toggle Options */}
              <div className="flex gap-3">
                <button
                  onClick={() => onOptionsChange("ref_acceleration", !tryOnOptions.ref_acceleration)}
                  className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-sm font-medium transition-all ${
                    tryOnOptions.ref_acceleration
                      ? "bg-amber-500 text-white"
                      : "bg-white text-gray-600 hover:bg-gray-100 border border-gray-200"
                  }`}
                >
                  <Zap className="h-4 w-4" />
                  Turbo
                </button>
                <button
                  onClick={() => onOptionsChange("repaint", !tryOnOptions.repaint)}
                  className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-sm font-medium transition-all ${
                    tryOnOptions.repaint
                      ? "bg-purple-500 text-white"
                      : "bg-white text-gray-600 hover:bg-gray-100 border border-gray-200"
                  }`}
                >
                  <Paintbrush className="h-4 w-4" />
                  Repaint
                </button>
              </div>
              <p className="text-xs text-gray-400">
                Turbo: Faster processing. Repaint: Better edge blending.
              </p>
            </div>
          )}
        </div>

        {/* Garment Grid */}
        <div>
          <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-3 sm:mb-4">
            Your Garments
          </h2>
          {isLoading ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 sm:gap-4">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="aspect-square bg-gray-200 rounded-lg animate-pulse" />
              ))}
            </div>
          ) : garments.length === 0 ? (
            <div className="text-center py-8 sm:py-12 text-gray-500 text-sm sm:text-base">
              <p>No garments yet. Upload your first garment above!</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 sm:gap-4">
              {garments.map((garment) => (
                <button
                  key={garment.id}
                  onClick={() => onGarmentSelect(garment)}
                  className={`
                    relative aspect-square rounded-lg overflow-hidden transition-all min-h-[44px]
                    ${selectedGarment?.id === garment.id
                      ? 'ring-4 ring-black shadow-lg'
                      : 'ring-1 ring-gray-200 hover:ring-2 hover:ring-gray-400'
                    }
                  `}
                  aria-label={`Select ${garment.name}${selectedGarment?.id === garment.id ? ' (currently selected)' : ''}`}
                  aria-pressed={selectedGarment?.id === garment.id}
                >
                  <Image
                    src={garment.thumbnailUrl || garment.url}
                    alt={`Garment: ${garment.name}, uploaded ${new Date(garment.uploadedAt).toLocaleDateString()}`}
                    fill
                    className="object-cover"
                    unoptimized
                  />
                  {selectedGarment?.id === garment.id && (
                    <div className="absolute top-2 right-2 bg-black text-white rounded-full p-1" aria-hidden="true">
                      <Check className="h-4 w-4" />
                    </div>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

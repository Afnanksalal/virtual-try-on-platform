"use client";

import { useState, useRef } from "react";
import Image from "next/image";
import { Upload, Check } from "lucide-react";
import { Garment } from "@/lib/types";
import { validateGarmentImage } from "@/lib/validation";
import { handleError, showSuccess, showInfo } from "@/lib/errorHandling";

interface GarmentTabProps {
  garments: Garment[];
  selectedGarment: Garment | null;
  onGarmentSelect: (garment: Garment) => void;
  onGarmentUpload: (file: File) => Promise<void>;
  isLoading?: boolean;
}

export default function GarmentTab({
  garments,
  selectedGarment,
  onGarmentSelect,
  onGarmentUpload,
  isLoading = false
}: GarmentTabProps) {
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
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

"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
import { Upload, User } from "lucide-react";
import { supabase } from "@/lib/supabase";
import { endpoints } from "@/lib/api";
import { validatePersonalImage } from "@/lib/validation";
import { handleError, showSuccess, showInfo } from "@/lib/errorHandling";

interface PersonalImageTabProps {
  personalImage?: {
    url: string;
    type: 'head-only' | 'full-body';
    uploadedAt: Date;
  } | null;
  onImageUpload?: (file: File) => Promise<void>;
  isLoading?: boolean;
}

export default function PersonalImageTab({ 
  personalImage: propPersonalImage, 
  onImageUpload: propOnImageUpload,
  isLoading: propIsLoading = false 
}: PersonalImageTabProps) {
  const [personalImage, setPersonalImage] = useState<{
    url: string;
    type: 'head-only' | 'full-body';
    uploadedAt: Date;
  } | null>(propPersonalImage || null);
  const [isLoading, setIsLoading] = useState(propIsLoading);
  const [uploading, setUploading] = useState(false);

  // Fetch personal image on mount if not provided via props
  useEffect(() => {
    if (!propPersonalImage) {
      fetchPersonalImage();
    }
  }, [propPersonalImage]);

  const fetchPersonalImage = async () => {
    try {
      setIsLoading(true);
      const { data: { session } } = await supabase.auth.getSession();
      
      if (!session) {
        setPersonalImage(null);
        return;
      }

      const result = await endpoints.getPersonalImage(session.user.id);
      setPersonalImage(result);
    } catch (error) {
      handleError(error, 'fetch personal image', { showToast: true });
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file
    const validationResult = await validatePersonalImage(file);
    if (!validationResult.isValid) {
      handleError(
        new Error(validationResult.error),
        'validate personal image',
        { showToast: true }
      );
      return;
    }

    try {
      setUploading(true);
      showInfo('Uploading personal image...');

      // Use prop handler if provided, otherwise use built-in handler
      if (propOnImageUpload) {
        await propOnImageUpload(file);
      } else {
        // Built-in upload handler
        const { data: { session } } = await supabase.auth.getSession();
        
        if (!session) {
          handleError(
            new Error('Not authenticated'),
            'upload personal image',
            { showToast: true }
          );
          return;
        }

        const result = await endpoints.updatePersonalImage(session.user.id, file);
        setPersonalImage(result);
      }

      showSuccess('Personal image updated successfully');
    } catch (error) {
      handleError(error, 'upload personal image', { showToast: true });
    } finally {
      setUploading(false);
    }
  };

  if (isLoading) {
    return (
      <div 
        role="tabpanel" 
        id="personal-panel" 
        aria-labelledby="personal-tab"
        className="bg-white rounded-b-xl p-4 sm:p-6 md:p-8"
      >
        <div className="flex items-center justify-center h-64 sm:h-80 md:h-96">
          <div className="animate-pulse text-gray-400">Loading...</div>
        </div>
      </div>
    );
  }

  if (!personalImage) {
    return (
      <div 
        role="tabpanel" 
        id="personal-panel" 
        aria-labelledby="personal-tab"
        className="bg-white rounded-b-xl p-4 sm:p-6 md:p-8"
      >
        <div className="flex flex-col items-center justify-center h-64 sm:h-80 md:h-96 text-center px-4">
          <div className="h-20 w-20 sm:h-24 sm:w-24 bg-gray-100 rounded-full flex items-center justify-center mb-4">
            <User className="h-10 w-10 sm:h-12 sm:w-12 text-gray-400" />
          </div>
          <h3 className="text-base sm:text-lg font-semibold text-gray-900 mb-2">
            No Personal Image
          </h3>
          <p className="text-sm sm:text-base text-gray-600 mb-6 max-w-md">
            You haven't uploaded a personal image yet. Complete onboarding to get started.
          </p>
          <a
            href="/onboard"
            className="bg-black text-white px-6 py-3 rounded-full font-medium hover:bg-gray-800 transition-colors min-h-[44px] flex items-center"
          >
            Complete Onboarding
          </a>
        </div>
      </div>
    );
  }

  return (
    <div 
      role="tabpanel" 
      id="personal-panel" 
      aria-labelledby="personal-tab"
      className="bg-white rounded-b-xl p-4 sm:p-6 md:p-8"
    >
      <div className="max-w-2xl mx-auto">
        <div className="mb-4 sm:mb-6">
          <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-2">
            Your Personal Image
          </h2>
          <p className="text-sm sm:text-base text-gray-600">
            This image will be used for virtual try-on
          </p>
        </div>

        <div className="bg-gray-50 rounded-xl p-4 sm:p-6 mb-4 sm:mb-6">
          <div className="relative aspect-[3/4] max-w-md mx-auto mb-4 bg-white rounded-lg overflow-hidden shadow-sm">
            <Image
              src={personalImage.url}
              alt={`Your ${personalImage.type === 'head-only' ? 'head-only' : 'full-body'} personal photo uploaded on ${new Date(personalImage.uploadedAt).toLocaleDateString()}`}
              fill
              className="object-cover"
              unoptimized
            />
          </div>

          <div className="flex flex-col sm:flex-row gap-2 sm:gap-4 text-xs sm:text-sm text-gray-600">
            <div className="flex-1">
              <span className="font-medium text-gray-900">Type:</span>{' '}
              <span className="capitalize">{personalImage.type.replace('-', ' ')}</span>
            </div>
            <div className="flex-1">
              <span className="font-medium text-gray-900">Uploaded:</span>{' '}
              {new Date(personalImage.uploadedAt).toLocaleDateString()}
            </div>
          </div>
        </div>

        <div className="flex justify-center">
          <label className="cursor-pointer">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
              disabled={uploading}
              aria-label="Upload new personal image"
            />
            <div className="flex items-center gap-2 bg-gray-900 text-white px-6 py-3 rounded-full font-medium hover:bg-gray-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed min-h-[44px] text-sm sm:text-base">
              <Upload className="h-4 w-4" aria-hidden="true" />
              {uploading ? 'Uploading...' : 'Re-upload Image'}
            </div>
          </label>
        </div>
      </div>
    </div>
  );
}

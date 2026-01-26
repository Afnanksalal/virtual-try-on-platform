"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowLeft, Upload, Trash2, X, Grid3x3, List, Calendar } from "lucide-react";
import { endpoints } from "@/lib/api";
import { validateGarmentImage } from "@/lib/validation";
import { handleError, showSuccess, showInfo } from "@/lib/errorHandling";
import type { Garment } from "@/lib/types";
import ProtectedRoute from "@/components/ProtectedRoute";
import { StudioProvider, useStudio } from "@/contexts/StudioContext";

type ViewMode = 'grid' | 'list';

function WardrobeContent() {
  const router = useRouter();
  const { garments, userId, addGarment, removeGarment, refreshGarments } = useStudio();
  const [selectedGarment, setSelectedGarment] = useState<Garment | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    // Garments are already loaded by StudioContext
    setIsLoading(false);
  }, []);

  const handleUploadGarment = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file
    const validationResult = await validateGarmentImage(file);
    if (!validationResult.isValid) {
      handleError(
        new Error(validationResult.error),
        'validate garment image',
        { showToast: true, userId: userId || undefined }
      );
      return;
    }

    if (!userId) {
      handleError(
        new Error('Not authenticated'),
        'upload garment',
        { showToast: true }
      );
      return;
    }

    // Optimistic update
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

    try {
      setIsUploading(true);
      addGarment(tempGarment); // Optimistic update
      showInfo("Uploading garment...");

      const newGarment = await endpoints.uploadGarment(file, userId);
      
      // Replace temp with real garment
      removeGarment(tempId);
      addGarment(newGarment);
      
      showSuccess("Garment uploaded successfully");
      event.target.value = ''; // Reset input
      URL.revokeObjectURL(tempUrl);
    } catch (error) {
      handleError(error, 'upload garment', { showToast: true, userId: userId || undefined });
      removeGarment(tempId);
      URL.revokeObjectURL(tempUrl);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDeleteGarment = async (garmentId: string) => {
    if (!userId) return;

    try {
      // Optimistic update
      const garmentToDelete = garments.find(g => g.id === garmentId);
      removeGarment(garmentId);
      setSelectedGarment(null);
      setDeleteConfirmId(null);
      showInfo("Deleting garment...");

      await endpoints.deleteGarment(garmentId, userId);
      showSuccess("Garment deleted successfully");
    } catch (error) {
      handleError(error, 'delete garment', { showToast: true, userId: userId || undefined });
      // Refresh to restore state on error
      await refreshGarments();
    }
  };

  const handleUseInStudio = (garment: Garment) => {
    // Store selected garment in sessionStorage for Studio to pick up
    sessionStorage.setItem('selectedGarment', JSON.stringify(garment));
    router.push('/studio');
  };

  const handleGarmentClick = (garment: Garment) => {
    setSelectedGarment(garment);
  };

  const closeDetailView = () => {
    setSelectedGarment(null);
  };

  return (
    <div className="min-h-screen pt-28 pb-12 px-4 bg-gray-50">
      <div className="max-w-7xl mx-auto">
          {/* Header */}
          <header className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-10">
            <div>
              <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-gray-900 mb-2">
                My Wardrobe
              </h1>
              <p className="text-sm sm:text-base text-gray-500">
                Your collection of garments for virtual try-on
              </p>
            </div>
            <div className="flex items-center gap-3">
              {/* View Mode Toggle */}
              <div className="flex items-center gap-1 bg-white rounded-lg p-1 shadow-sm">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`p-2 rounded transition-colors ${
                    viewMode === 'grid' 
                      ? 'bg-gray-900 text-white' 
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                  aria-label="Grid view"
                >
                  <Grid3x3 className="h-4 w-4" />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`p-2 rounded transition-colors ${
                    viewMode === 'list' 
                      ? 'bg-gray-900 text-white' 
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                  aria-label="List view"
                >
                  <List className="h-4 w-4" />
                </button>
              </div>
              
              <Link 
                href="/studio" 
                className="flex items-center gap-2 bg-black text-white px-4 py-2 rounded-xl text-sm font-medium hover:bg-gray-800 transition-colors"
              >
                <ArrowLeft className="h-4 w-4" />
                Back to Studio
              </Link>
            </div>
          </header>

          {/* Loading State */}
          {isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="bg-white rounded-3xl overflow-hidden shadow-sm animate-pulse">
                  <div className="aspect-3/4 bg-gray-200" />
                  <div className="p-4 space-y-2">
                    <div className="h-4 bg-gray-200 rounded w-3/4" />
                    <div className="h-3 bg-gray-200 rounded w-1/2" />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <>
              {/* Empty State */}
              {garments.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-20 text-center">
                  <div className="bg-gray-100 p-6 rounded-full mb-6">
                    <Upload className="h-12 w-12 text-gray-400" />
                  </div>
                  <h2 className="text-2xl font-bold text-gray-900 mb-2">
                    No garments yet
                  </h2>
                  <p className="text-gray-500 mb-6 max-w-md">
                    Upload your first garment to start building your virtual wardrobe
                  </p>
                  <label className="cursor-pointer bg-black text-white px-6 py-3 rounded-xl font-medium hover:bg-gray-800 transition-colors">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleUploadGarment}
                      className="hidden"
                      disabled={isUploading}
                    />
                    {isUploading ? 'Uploading...' : 'Upload Garment'}
                  </label>
                </div>
              ) : (
                <>
                  {/* Grid View */}
                  {viewMode === 'grid' && (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                      {/* Upload Card */}
                      <label className="group cursor-pointer border-2 border-dashed border-gray-200 rounded-3xl flex flex-col items-center justify-center min-h-[400px] hover:border-gray-400 transition-colors">
                        <input
                          type="file"
                          accept="image/*"
                          onChange={handleUploadGarment}
                          className="hidden"
                          disabled={isUploading}
                        />
                        <div className="bg-gray-100 p-4 rounded-full mb-4 group-hover:scale-110 transition-transform">
                          <Upload className="h-6 w-6 text-gray-600" />
                        </div>
                        <span className="font-medium text-gray-600">
                          {isUploading ? 'Uploading...' : 'Upload Garment'}
                        </span>
                      </label>

                      {/* Garment Cards */}
                      {garments.map((garment, i) => (
                        <motion.div
                          key={garment.id}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: i * 0.05 }}
                          className="group relative bg-white rounded-3xl overflow-hidden shadow-sm hover:shadow-md transition-all cursor-pointer"
                          onClick={() => handleGarmentClick(garment)}
                          role="button"
                          tabIndex={0}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault();
                              handleGarmentClick(garment);
                            }
                          }}
                          aria-label={`View details for ${garment.name}, uploaded on ${garment.uploadedAt.toLocaleDateString()}`}
                        >
                          <div className="relative aspect-3/4">
                            <Image
                              src={garment.thumbnailUrl}
                              alt={`${garment.name} - garment in your wardrobe`}
                              fill
                              className="object-cover"
                            />
                            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors" />
                          </div>
                          <div className="p-4">
                            <h3 className="font-medium text-gray-900 truncate mb-1">
                              {garment.name}
                            </h3>
                            <div className="flex items-center gap-2 text-xs text-gray-500">
                              <Calendar className="h-3 w-3" aria-hidden="true" />
                              {garment.uploadedAt.toLocaleDateString()}
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  )}

                  {/* List View */}
                  {viewMode === 'list' && (
                    <div className="space-y-4">
                      {/* Upload Button */}
                      <label className="cursor-pointer flex items-center gap-4 p-4 bg-white rounded-2xl border-2 border-dashed border-gray-200 hover:border-gray-400 transition-colors">
                        <input
                          type="file"
                          accept="image/*"
                          onChange={handleUploadGarment}
                          className="hidden"
                          disabled={isUploading}
                        />
                        <div className="bg-gray-100 p-3 rounded-xl">
                          <Upload className="h-5 w-5 text-gray-600" />
                        </div>
                        <span className="font-medium text-gray-600">
                          {isUploading ? 'Uploading...' : 'Upload New Garment'}
                        </span>
                      </label>

                      {/* Garment List Items */}
                      {garments.map((garment, i) => (
                        <motion.div
                          key={garment.id}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.05 }}
                          className="flex items-center gap-4 p-4 bg-white rounded-2xl shadow-sm hover:shadow-md transition-all cursor-pointer"
                          onClick={() => handleGarmentClick(garment)}
                        >
                          <div className="relative w-24 h-24 rounded-xl overflow-hidden flex-shrink-0">
                            <Image
                              src={garment.thumbnailUrl}
                              alt={garment.name}
                              fill
                              className="object-cover"
                            />
                          </div>
                          <div className="flex-1 min-w-0">
                            <h3 className="font-medium text-gray-900 truncate mb-1">
                              {garment.name}
                            </h3>
                            <div className="flex items-center gap-2 text-sm text-gray-500">
                              <Calendar className="h-3 w-3" />
                              {garment.uploadedAt.toLocaleDateString()}
                            </div>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </>
          )}

          {/* Detail View Modal */}
          <AnimatePresence>
            {selectedGarment && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
                onClick={closeDetailView}
              >
                <motion.div
                  initial={{ scale: 0.9, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0.9, opacity: 0 }}
                  className="bg-white rounded-3xl max-w-2xl w-full max-h-[90vh] overflow-y-auto"
                  onClick={(e) => e.stopPropagation()}
                >
                  {/* Close Button */}
                  <div className="sticky top-0 bg-white border-b border-gray-100 p-4 flex justify-between items-center">
                    <h2 className="text-xl font-bold text-gray-900">Garment Details</h2>
                    <button
                      onClick={closeDetailView}
                      className="p-2 hover:bg-gray-100 rounded-full transition-colors"
                    >
                      <X className="h-5 w-5" />
                    </button>
                  </div>

                  {/* Image */}
                  <div className="relative aspect-3/4 bg-gray-100">
                    <Image
                      src={selectedGarment.url}
                      alt={`Full view of ${selectedGarment.name} - ${(selectedGarment.metadata.size / 1024 / 1024).toFixed(2)} MB ${selectedGarment.metadata.format || 'image'}`}
                      fill
                      className="object-contain"
                    />
                  </div>

                  {/* Metadata */}
                  <div className="p-6 space-y-4">
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900 mb-2">
                        {selectedGarment.name}
                      </h3>
                      <div className="space-y-2 text-sm text-gray-600">
                        <div className="flex justify-between">
                          <span>Uploaded:</span>
                          <span className="font-medium">
                            {selectedGarment.uploadedAt.toLocaleDateString()}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Size:</span>
                          <span className="font-medium">
                            {(selectedGarment.metadata.size / 1024 / 1024).toFixed(2)} MB
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Format:</span>
                          <span className="font-medium">
                            {selectedGarment.metadata.format || 'Image'}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex gap-3 pt-4 border-t border-gray-100">
                      <button
                        onClick={() => handleUseInStudio(selectedGarment)}
                        className="flex-1 bg-black text-white px-6 py-3 rounded-xl font-medium hover:bg-gray-800 transition-colors"
                        aria-label={`Use ${selectedGarment.name} in virtual try-on studio`}
                      >
                        Use in Studio
                      </button>
                      
                      {deleteConfirmId === selectedGarment.id ? (
                        <div className="flex-1 flex gap-2">
                          <button
                            onClick={() => handleDeleteGarment(selectedGarment.id)}
                            className="flex-1 bg-red-600 text-white px-4 py-3 rounded-xl font-medium hover:bg-red-700 transition-colors"
                            aria-label={`Confirm deletion of ${selectedGarment.name}`}
                          >
                            Confirm
                          </button>
                          <button
                            onClick={() => setDeleteConfirmId(null)}
                            className="flex-1 bg-gray-200 text-gray-700 px-4 py-3 rounded-xl font-medium hover:bg-gray-300 transition-colors"
                            aria-label="Cancel deletion"
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <button
                          onClick={() => setDeleteConfirmId(selectedGarment.id)}
                          className="px-6 py-3 bg-red-50 text-red-600 rounded-xl font-medium hover:bg-red-100 transition-colors flex items-center gap-2"
                          aria-label={`Delete ${selectedGarment.name} from wardrobe`}
                        >
                          <Trash2 className="h-4 w-4" aria-hidden="true" />
                          Delete
                        </button>
                      )}
                    </div>
                  </div>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
      </div>
    </div>
  );
}

export default function WardrobePage() {
  return (
    <ProtectedRoute>
      <StudioProvider>
        <WardrobeContent />
      </StudioProvider>
    </ProtectedRoute>
  );
}

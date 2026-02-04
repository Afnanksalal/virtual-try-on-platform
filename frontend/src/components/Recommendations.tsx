"use client";
import { Tag, Loader2, ShoppingBag } from "lucide-react";
import { useEffect, useState } from "react";
import Image from "next/image";
import { endpoints, APIError } from "@/lib/api";
import { toast } from "sonner";
import { supabase } from "@/lib/supabase";

interface Recommendation {
  id: string;
  name: string;
  image_url: string;
  price: number;
  currency: string;
  category: string;
  ebay_url: string;
}

// Placeholder image as data URI - no external dependencies
const PLACEHOLDER_IMAGE = "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjQwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjNmNGY2Ii8+PHRleHQgeD0iNTAlIiB5PSI0NSUiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiBmb250LXNpemU9IjQ4IiBmaWxsPSIjOWNhM2FmIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj7wn5GVPC90ZXh0Pjx0ZXh0IHg9IjUwJSIgeT0iNTUlIiBmb250LWZhbWlseT0ic2Fucy1zZXJpZiIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzljYTNhZiIgdGV4dC1hbmNob3I9Im1pZGRsZSI+SW1hZ2UgVW5hdmFpbGFibGU8L3RleHQ+PC9zdmc+";

export default function Recommendations() {
  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState<Recommendation | null>(null);
  const [failedImages, setFailedImages] = useState<Set<string>>(new Set());

  const handleImageError = (id: string) => {
    console.warn(`[Recommendations] Image failed to load for item: ${id}`);
    setFailedImages(prev => new Set([...prev, id]));
  };

export default function Recommendations() {
  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState<Recommendation | null>(null);

  useEffect(() => {
    loadRecommendations();
  }, []);

  const loadRecommendations = async () => {
    setLoading(true);
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        toast.error("Please log in to get recommendations");
        return;
      }

      const { data: profile } = await supabase
        .from('profiles')
        .select('photo_url')
        .eq('id', session.user.id)
        .single();

      if (!profile?.photo_url) {
        toast.error("Please complete your profile first");
        return;
      }

      const photoResponse = await fetch(profile.photo_url);
      const photoBlob = await photoResponse.blob();
      const photoFile = new File([photoBlob], 'user_photo.jpg', { type: 'image/jpeg' });

      const recommendations = await endpoints.getRecommendations(photoFile);
      setRecs(recommendations);
      
    } catch (error) {
      console.error("Failed to load recommendations:", error);
      if (error instanceof APIError) {
        toast.error(error.message);
      } else {
        toast.error("Failed to load recommendations");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleBuyClick = (product: Recommendation) => {
    setSelectedProduct(product);
    setShowModal(true);
  };

  const handleProceed = () => {
    if (selectedProduct) {
      window.open(selectedProduct.ebay_url, '_blank');
      setShowModal(false);
      setSelectedProduct(null);
    }
  };

  const handleCancel = () => {
    setShowModal(false);
    setSelectedProduct(null);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="animate-spin h-8 w-8 text-primary-600" />
      </div>
    );
  }

  return (
    <>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h3 className="text-xl font-bold text-gray-900">AI-Powered Recommendations</h3>
          <Tag className="h-5 w-5 text-primary-500" />
        </div>

        {recs.length === 0 ? (
          <p className="text-gray-500 text-sm">No recommendations available. Complete your profile to get personalized suggestions.</p>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {recs.map((rec, i) => (
              <div key={rec.id || i} className="bg-white border border-gray-200 rounded-2xl p-4 hover:shadow-md transition-shadow">
                <div className="aspect-square bg-gray-100 rounded-xl mb-3 overflow-hidden relative">
                  <Image 
                    src={failedImages.has(rec.id) ? PLACEHOLDER_IMAGE : rec.image_url} 
                    alt={rec.name}
                    fill
                    sizes="(max-width: 768px) 100vw, (max-width: 1024px) 50vw, 33vw"
                    className="object-cover"
                    unoptimized
                    onError={() => handleImageError(rec.id)}
                  />
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h4 className="font-semibold text-gray-900 text-sm line-clamp-1">{rec.name}</h4>
                    <span className="text-sm font-medium text-primary-600">{rec.currency} ${rec.price.toFixed(2)}</span>
                  </div>
                  <p className="text-xs text-gray-500 uppercase tracking-wide">{rec.category}</p>
                  <button
                    onClick={() => handleBuyClick(rec)}
                    className="flex items-center justify-center gap-2 w-full bg-black text-white py-2 px-4 rounded-lg text-sm font-medium hover:bg-gray-800 transition-colors"
                  >
                    <ShoppingBag className="h-4 w-4" />
                    Buy on eBay
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* eBay Redirect Modal */}
      {showModal && selectedProduct && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={handleCancel}>
          <div className="bg-white rounded-2xl p-6 max-w-md w-full shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-xl font-bold text-gray-900 mb-2">Redirecting to eBay</h3>
            <p className="text-gray-600 mb-6">
              You will be redirected to eBay to purchase <strong>{selectedProduct.name}</strong> for <strong>${selectedProduct.price.toFixed(2)}</strong>.
            </p>
            <div className="flex gap-3">
              <button
                onClick={handleCancel}
                className="flex-1 px-4 py-3 border-2 border-gray-300 text-gray-700 rounded-xl font-semibold hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleProceed}
                className="flex-1 px-4 py-3 bg-black text-white rounded-xl font-semibold hover:bg-gray-800 transition-colors"
              >
                Proceed to eBay
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

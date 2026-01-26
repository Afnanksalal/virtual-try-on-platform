"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { Sparkles, Loader2, ShoppingBag, RefreshCw } from "lucide-react";
import { supabase } from "@/lib/supabase";
import { endpoints } from "@/lib/api";
import { toast } from "sonner";
import ProtectedRoute from "@/components/ProtectedRoute";

interface Recommendation {
  id: string;
  name: string;
  image_url: string;
  price: number;
  currency: string;
  category: string;
  ebay_url: string;
}

export default function ShopPage() {
  const router = useRouter();
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadRecommendations();
  }, []);

  const loadRecommendations = async () => {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        router.push('/auth');
        return;
      }

      const { data: profile } = await supabase
        .from('profiles')
        .select('photo_url')
        .eq('id', session.user.id)
        .single();

      if (!profile?.photo_url) {
        toast.error("Please complete your profile first");
        router.push('/onboard');
        return;
      }

      const photoResponse = await fetch(profile.photo_url);
      const photoBlob = await photoResponse.blob();
      const photoFile = new File([photoBlob], 'user_photo.jpg', { type: 'image/jpeg' });

      const recs = await endpoints.getRecommendations(photoFile);
      setRecommendations(recs);
    } catch (error) {
      console.error("Failed to load recommendations:", error);
      toast.error("Failed to load recommendations");
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadRecommendations();
    setRefreshing(false);
    toast.success("Recommendations refreshed");
  };

  const handleBuyClick = (rec: Recommendation) => {
    window.open(rec.ebay_url, '_blank');
  };

  if (loading) {
    return (
      <ProtectedRoute>
        <div className="min-h-screen pt-28 pb-12 px-4 bg-gray-50 flex items-center justify-center">
          <Loader2 className="animate-spin h-12 w-12 text-gray-400" />
        </div>
      </ProtectedRoute>
    );
  }

  return (
    <ProtectedRoute>
      <div className="min-h-screen pt-28 pb-12 px-4 bg-gray-50">
        <div className="max-w-7xl mx-auto">
          <div className="mb-6 sm:mb-8 flex items-center justify-between">
            <div>
              <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-gray-900 mb-2">
                AI Recommendations
              </h1>
              <p className="text-sm sm:text-base text-gray-600">
                Personalized outfit suggestions powered by AI
              </p>
            </div>
            <button
              onClick={handleRefresh}
              disabled={refreshing}
              className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-200 rounded-xl hover:bg-gray-50 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
              <span className="hidden sm:inline">Refresh</span>
            </button>
          </div>

          {recommendations.length === 0 ? (
            <div className="bg-white rounded-3xl p-12 text-center shadow-sm">
              <Sparkles className="h-20 w-20 text-gray-300 mx-auto mb-6" />
              <h2 className="text-2xl font-bold text-gray-900 mb-4">No Recommendations Yet</h2>
              <p className="text-gray-600 mb-8 max-w-2xl mx-auto">
                Complete your profile to get personalized outfit recommendations from eBay
              </p>
              <button
                onClick={() => router.push('/onboard')}
                className="bg-black text-white px-8 py-4 rounded-xl font-semibold hover:bg-gray-800 transition-colors"
              >
                Complete Profile
              </button>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {recommendations.map((rec) => (
                <div key={rec.id} className="bg-white rounded-2xl overflow-hidden shadow-sm hover:shadow-md transition-shadow">
                  <div className="relative aspect-square bg-gray-100">
                    <Image 
                      src={rec.image_url} 
                      alt={rec.name}
                      fill
                      className="object-cover"
                      unoptimized
                    />
                  </div>
                  <div className="p-4">
                    <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">{rec.category}</p>
                    <h3 className="font-semibold text-gray-900 mb-2 line-clamp-2">{rec.name}</h3>
                    <div className="flex items-center justify-between">
                      <p className="text-lg font-bold text-gray-900">
                        {rec.currency === 'USD' ? '$' : rec.currency}{rec.price.toFixed(2)}
                      </p>
                      <button 
                        onClick={() => handleBuyClick(rec)}
                        className="bg-black text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-gray-800 transition-colors flex items-center gap-2"
                      >
                        <ShoppingBag className="w-4 h-4" />
                        Buy
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </ProtectedRoute>
  );
}

"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import Image from "next/image";
import { motion } from "framer-motion";
import { ArrowLeft, Clock, Shirt } from "lucide-react";
import { toast } from "sonner";
import { supabase } from "@/lib/supabase";
import ProtectedRoute from "@/components/ProtectedRoute";

type HistoryItem = {
  id: string;
  cloth_image_url: string;
  result_image_url: string;
  created_at: string;
};

export default function WardrobePage() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const userId = localStorage.getItem("vton_user_id");
        if (!userId) {
          toast.error("Please log in first");
          setLoading(false);
          return;
        }

        setLoading(true);

        // Fetch history directly from Supabase
        const { data, error } = await supabase
          .from('history')
          .select('*')
          .eq('user_id', userId)
          .order('created_at', { ascending: false });

        if (error) throw error;

        setHistory(data || []);
      } catch (error) {
        console.error("Failed to load history:", error);
        toast.error("Failed to load your wardrobe");
      } finally {
        setLoading(false);
      }
    };

    loadHistory();
  }, []);

  return (
    <ProtectedRoute>
      <div className="min-h-screen pt-28 pb-12 px-4 bg-gray-50">
        <div className="max-w-7xl mx-auto">
          <header className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 sm:gap-0 mb-10">
            <div>
              <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-gray-900 mb-2">My Wardrobe</h1>
              <p className="text-sm sm:text-base text-gray-500">Your digital closet and past styles</p>
            </div>
            <Link href="/studio" className="flex items-center gap-2 bg-black text-white px-4 py-2 rounded-xl text-sm font-medium hover:bg-gray-800 transition-colors">
              <ArrowLeft className="h-4 w-4" />
              Back to Studio
            </Link>
          </header>

          {loading ? (
            <div className="flex items-center justify-center py-20">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {/* Create New Card */}
              <Link href="/studio" className="group border-2 border-dashed border-gray-200 rounded-3xl flex flex-col items-center justify-center min-h-[400px] hover:border-gray-400 transition-colors">
                <div className="bg-gray-100 p-4 rounded-full mb-4 group-hover:scale-110 transition-transform">
                  <Shirt className="h-6 w-6 text-gray-600" />
                </div>
                <span className="font-medium text-gray-600">Create New Look</span>
              </Link>

              {history.map((item, i) => (
                <motion.div 
                  key={item.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className="group relative bg-white rounded-3xl overflow-hidden shadow-sm hover:shadow-md transition-all"
                >
                  <div className="relative aspect-3/4">
                    <Image 
                      src={item.result_image_url} 
                      alt="Try On Result" 
                      fill 
                      className="object-cover" 
                    />
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors" />
                  </div>
                  <div className="absolute bottom-0 left-0 w-full p-4 bg-white/90 backdrop-blur transform translate-y-full group-hover:translate-y-0 transition-transform">
                    <div className="flex items-center gap-2 text-xs text-gray-500">
                      <Clock className="h-3 w-3" />
                      {new Date(item.created_at).toLocaleDateString()}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </div>
    </ProtectedRoute>
  );
}

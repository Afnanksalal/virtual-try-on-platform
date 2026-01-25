"use client";
import { useState } from "react";
import { Upload, Shirt, Wand2 } from "lucide-react";
import Image from "next/image";
import { toast } from "sonner";
import { endpoints, APIError } from "@/lib/api";
import { supabase } from "@/lib/supabase";

export default function TryOnWidget({ onResult }: { onResult: (url: string) => void }) {
  const [cloth, setCloth] = useState<string | null>(null);
  const [clothFile, setClothFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);

  const handleRun = async () => {
      if (!clothFile) return;
      setLoading(true);
      
      try {
          const userId = localStorage.getItem("vton_user_id");
          if (!userId) throw new Error("User session not found. Please log in.");

          // Get user's photo from Supabase
          const { data: { session } } = await supabase.auth.getSession();
          if (!session) throw new Error("Please log in first");
          
          const { data: profile } = await supabase
            .from('profiles')
            .select('photo_url')
            .eq('id', session.user.id)
            .single();

          if (!profile?.photo_url) {
            throw new Error("Please complete your profile with a photo first");
          }

          // Fetch user image from URL
          const userImageResponse = await fetch(profile.photo_url);
          const userImageBlob = await userImageResponse.blob();
          const userImageFile = new File([userImageBlob], "user.jpg", { type: "image/jpeg" });

          // Call ML API with both images
          const data = await endpoints.processTryOn(userImageFile, clothFile);
          
          onResult(data.result_url); 
          toast.success(`Try-on completed in ${data.processing_time}s!`);
      } catch (error) {
          console.error("Try-on error:", error);
          const msg = error instanceof APIError ? error.message : "Try-on failed. Please try again.";
          toast.error(msg);
      } finally {
          setLoading(false);
      }
  };

  return (
    <div className="bg-white rounded-3xl p-6 shadow-sm border border-gray-100 h-full flex flex-col">
       <h3 className="text-lg font-bold mb-6 flex items-center gap-2">
           <Shirt className="h-5 w-5 text-primary-500" />
           Select Apparel
       </h3>
       
       <div className="flex-1 border-2 border-dashed border-gray-200 rounded-2xl flex flex-col items-center justify-center bg-gray-50 hover:bg-white transition-colors cursor-pointer group mb-6 relative overflow-hidden">
           {cloth ? (
               <Image src={cloth} alt="Selected Cloth" fill className="object-cover" />
           ) : (
               <div className="text-center p-6">
                   <Upload className="h-8 w-8 text-gray-400 mx-auto mb-3 group-hover:scale-110 transition-transform" />
                   <p className="text-sm font-medium text-gray-600">Upload Cloth Image</p>
               </div>
           )}
           <input 
             type="file" 
             accept="image/png, image/jpeg, image/webp"
             className="absolute inset-0 opacity-0 cursor-pointer"
             onChange={(e) => {
                 const file = e.target.files?.[0];
                 if(file) {
                    if (file.size > 10 * 1024 * 1024) {
                        alert("File size must be under 10MB");
                        return;
                    }
                    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
                        alert("Only JPG, PNG, WEBP files allowed");
                        return;
                    }

                    setClothFile(file);
                    setCloth(URL.createObjectURL(file));
                 }
             }}
           />
       </div>
       
       <button
         onClick={handleRun}
         disabled={!cloth || loading}
         className="w-full bg-primary-600 text-white py-3 rounded-xl font-medium flex items-center justify-center gap-2 hover:bg-primary-700 disabled:opacity-50 transition-all"
       >
           {loading ? (
             <span className="animate-pulse">Synthesizing...</span>
           ) : (
             <>
               <Wand2 className="h-4 w-4" />
               Visualize Fit
             </>
           )}
       </button>
    </div>
  );
}

"use client";
import { useState } from "react";
import { Upload, Shirt, Wand2, ChevronDown, ChevronUp, Zap, Paintbrush } from "lucide-react";
import Image from "next/image";
import { toast } from "sonner";
import { endpoints, APIError } from "@/lib/api";
import { supabase } from "@/lib/supabase";
import type { GarmentType, ModelType, TryOnOptions } from "@/lib/types";

export default function TryOnWidget({ onResult }: { onResult: (url: string) => void }) {
  const [cloth, setCloth] = useState<string | null>(null);
  const [clothFile, setClothFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  // Advanced options state
  const [garmentType, setGarmentType] = useState<GarmentType>("upper_body");
  const [modelType, setModelType] = useState<ModelType>("viton_hd");
  const [steps, setSteps] = useState(30);
  const [guidanceScale, setGuidanceScale] = useState(2.5);
  const [refAcceleration, setRefAcceleration] = useState(false);
  const [repaint, setRepaint] = useState(false);

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

          // Build options with advanced settings
          const options: TryOnOptions = {
            garment_type: garmentType,
            model_type: modelType,
            num_inference_steps: steps,
            guidance_scale: guidanceScale,
            ref_acceleration: refAcceleration,
            repaint: repaint,
          };

          // Call ML API with both images and options
          const data = await endpoints.processTryOn(userImageFile, clothFile, options);
          
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
       <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
           <Shirt className="h-5 w-5 text-primary-500" />
           Select Apparel
       </h3>
       
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
               onClick={() => setGarmentType(option.value as GarmentType)}
               className={`py-2 px-3 rounded-lg text-sm font-medium transition-all ${
                 garmentType === option.value
                   ? "bg-primary-600 text-white"
                   : "bg-gray-100 text-gray-600 hover:bg-gray-200"
               }`}
             >
               {option.label}
             </button>
           ))}
         </div>
       </div>

       {/* Upload Area */}
       <div className="flex-1 min-h-[200px] border-2 border-dashed border-gray-200 rounded-2xl flex flex-col items-center justify-center bg-gray-50 hover:bg-white transition-colors cursor-pointer group mb-4 relative overflow-hidden">
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

       {/* Advanced Options Toggle */}
       <button
         onClick={() => setShowAdvanced(!showAdvanced)}
         className="w-full flex items-center justify-between py-2 px-3 mb-4 text-sm text-gray-600 hover:text-gray-800 transition-colors"
       >
         <span className="font-medium">Advanced Options</span>
         {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
       </button>

       {/* Advanced Options Panel */}
       {showAdvanced && (
         <div className="mb-4 p-4 bg-gray-50 rounded-xl space-y-4">
           {/* Model Type */}
           <div>
             <label className="text-sm font-medium text-gray-700 mb-2 block">Model</label>
             <div className="grid grid-cols-2 gap-2">
               <button
                 onClick={() => setModelType("viton_hd")}
                 className={`py-2 px-3 rounded-lg text-sm font-medium transition-all ${
                   modelType === "viton_hd"
                     ? "bg-primary-600 text-white"
                     : "bg-white text-gray-600 hover:bg-gray-100 border border-gray-200"
                 }`}
               >
                 VITON-HD
               </button>
               <button
                 onClick={() => setModelType("dress_code")}
                 className={`py-2 px-3 rounded-lg text-sm font-medium transition-all ${
                   modelType === "dress_code"
                     ? "bg-primary-600 text-white"
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
               <span className="text-sm text-gray-500">{steps}</span>
             </div>
             <input
               type="range"
               min="10"
               max="50"
               value={steps}
               onChange={(e) => setSteps(Number(e.target.value))}
               className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
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
               <span className="text-sm text-gray-500">{guidanceScale.toFixed(1)}</span>
             </div>
             <input
               type="range"
               min="1"
               max="5"
               step="0.1"
               value={guidanceScale}
               onChange={(e) => setGuidanceScale(Number(e.target.value))}
               className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
             />
             <div className="flex justify-between text-xs text-gray-400 mt-1">
               <span>Creative (1.0)</span>
               <span>Precise (5.0)</span>
             </div>
           </div>

           {/* Toggle Options */}
           <div className="flex gap-3">
             <button
               onClick={() => setRefAcceleration(!refAcceleration)}
               className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-sm font-medium transition-all ${
                 refAcceleration
                   ? "bg-amber-500 text-white"
                   : "bg-white text-gray-600 hover:bg-gray-100 border border-gray-200"
               }`}
             >
               <Zap className="h-4 w-4" />
               Turbo
             </button>
             <button
               onClick={() => setRepaint(!repaint)}
               className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 rounded-lg text-sm font-medium transition-all ${
                 repaint
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

"use client";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight, Sparkles, Loader2, Dumbbell, Users, Flame, User } from "lucide-react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";
import { toast } from "sonner";
import { endpoints } from "@/lib/api";
import Image from "next/image";

type Step = "info" | "photo" | "body_selection" | "body_generation";

export default function OnboardPage() {
  const router = useRouter();
  const [step, setStep] = useState<Step>("info");
  const [formData, setFormData] = useState({
    name: "",
    age: "",
    gender: "female",
    height: "",
    weight: "",
    ethnicity: "",
    file: null as File | null
  });
  const [loading, setLoading] = useState(false);
  const [photoType, setPhotoType] = useState<"head_only" | "full_body" | null>(null);
  const [generatedBodies, setGeneratedBodies] = useState<Array<{id: string; data: string}>>([]);
  const [selectedBody, setSelectedBody] = useState<string>("");
  const [composedImage, setComposedImage] = useState<string>("");

  const bodyTypes = [
    { id: "athletic", label: "Athletic", icon: "Dumbbell" },
    { id: "slim", label: "Slim", icon: "Users" },
    { id: "muscular", label: "Muscular", icon: "Flame" },
    { id: "average", label: "Average", icon: "User" },
  ];

  // Enforce Auth
  useEffect(() => {
    const checkSession = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        router.replace("/auth");
      }
    };
    checkSession();
  }, [router]);

  const handlePhotoUpload = async (file: File) => {
    setFormData({ ...formData, file });
    setLoading(true);
    
    try {
      // Analyze if head-only or full-body
      const analysis = await endpoints.analyzeImage(file);
      setPhotoType(analysis.type);
      
      if (analysis.type === "head_only") {
        toast.info("Head-only photo detected! Let's generate a body for you.");
        setStep("body_selection");
      } else {
        toast.success("Full-body photo detected!");
        // Can proceed directly
      }
    } catch (error) {
      console.error("Analysis error:", error);
      toast.error("Failed to analyze image. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleBodyTypeSelect = async (bodyType: string) => {
    setLoading(true);
    
    try {
      // Generate body options
      const result = await endpoints.generateBodies({
       ethnicity: formData.ethnicity || "Mixed",
        skin_tone: "Medium", // Could be extracted from photo or asked
        body_type: bodyType,
        height_cm: parseFloat(formData.height),
        weight_kg: parseFloat(formData.weight),
      });
      
      setGeneratedBodies(result.images);
      setStep("body_generation");
      toast.success("Body options generated!");
    } catch (error) {
      console.error("Body generation error:", error);
      toast.error("Failed to generate bodies. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const    handleBodySelect = async (bodyData: string) => {
    setSelectedBody(bodyData);
    setLoading(true);
    
    try {
      if (!formData.file) return;
      
      // Convert base64 to File
      const bodyFile = await fetch(bodyData).then(r => r.blob()).then(b => new File([b], "body.png"));
      
      // Combine head + body
      const result = await endpoints.combineHeadBody(formData.file, bodyFile);
      setComposedImage(result.image_data);
      
      toast.success("Images combined successfully!");
      // Proceed to save
      await saveProfile(result.image_data);
    } catch (error) {
      console.error("Combination error:", error);
      toast.error("Failed to combine images. Please try again.");
      setLoading(false);
    }
  };

  const saveProfile = async (photoDataUrl?: string) => {
    const {data: { session } } = await supabase.auth.getSession();
    if (!session) {
      toast.error("Session expired. Please log in again.");
      router.push("/auth");
      return;
    }

    setLoading(true);    
    try {
      let publicUrl = "";
      
      if (photoDataUrl) {
        // Upload composed image
        const blob = await fetch(photoDataUrl).then(r => r.blob());
        const file = new File([blob], `composed_${Date.now()}.png`);
        
        const fileName = `${session.user.id}/${Date.now()}_composed.png`;
        const { error: uploadError } = await supabase.storage
          .from('user-uploads')
          .upload(fileName, file, { cacheControl: '3600', upsert: false });

        if (uploadError) throw uploadError;

        const { data: { publicUrl: url } } = supabase.storage
          .from('user-uploads')
          .getPublicUrl(fileName);
        publicUrl = url;
      } else {
        // Upload original photo
        if (!formData.file) throw new Error("No photo");
        
        const fileName = `${session.user.id}/${Date.now()}_${formData.file.name}`;
        const { error: uploadError } = await supabase.storage
          .from('user-uploads')
          .upload(fileName, formData.file, { cacheControl: '3600', upsert: false });

        if (uploadError) throw uploadError;

        const { data: { publicUrl: url } } = supabase.storage
          .from('user-uploads')
          .getPublicUrl(fileName);
        publicUrl = url;
      }

      // Create profile
      const { error: profileError } = await supabase
        .from('profiles')
        .insert({
          id: session.user.id,
          email: session.user.email,
          name: formData.name.trim(),
          age: parseInt(formData.age),
          gender: formData.gender,
          height_cm: parseFloat(formData.height),
          weight_kg: parseFloat(formData.weight),
          photo_url: publicUrl,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        });

      if (profileError) throw profileError;
      
      localStorage.setItem("user_name", formData.name.trim());
      toast.success("Profile created successfully!");
      
      setTimeout(() => router.push("/studio"), 500);
    } catch (error) {
      console.error("Error:", error);
      const message = error instanceof Error ? error.message : "Failed to create profile";
      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = () => {
    // Validation
    if (!formData.name.trim()) return toast.error("Name is required");
    const age = parseInt(formData.age);
    if (isNaN(age) || age < 13 || age > 100) return toast.error("Please enter a valid age (13-100)");
    const height = parseFloat(formData.height);
    if (isNaN(height) || height < 50 || height > 300) return toast.error("Please enter a valid height in cm");
    const weight = parseFloat(formData.weight);
    if (isNaN(weight) || weight < 20 || weight > 500) return toast.error("Please enter a valid weight in kg");
    if (!formData.file) return toast.error("Please upload a photo");

    if (photoType === "head_only") {
      if (!selectedBody) return toast.error("Please select a body");
    }

    // Will be saved after body generation flow or directly
    if (photoType === "full_body") {
      saveProfile();
    } else if (photoType === "head_only" && composedImage) {
      saveProfile(composedImage);
    }
  };

  return (
    <div className="min-h-screen pt-24 pb-12 px-4 bg-gray-50">
      <div className="max-w-4xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-900 mb-2">Complete Your Profile</h1>
            <p className="text-gray-600">Let&apos;s get you set up for virtual try-on</p>
          </div>

          <div className="bg-white rounded-3xl p-8 shadow-sm">
            <AnimatePresence mode="wait">
              {step === "info" && (
                <motion.div key="info" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <div className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <input
                        type="text"
                        placeholder="Full Name"
                        value={formData.name}
                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                        className="px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-black focus:border-transparent"
                      />
                      <input
                        type="number"
                        placeholder="Age"
                        value={formData.age}
                        onChange={(e) => setFormData({ ...formData, age: e.target.value })}
                        className="px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-black focus:border-transparent"
                      />
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <select
                        value={formData.gender}
                        onChange={(e) => setFormData({ ...formData, gender: e.target.value })}
                        className="px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-black focus:border-transparent"
                      >
                        <option value="female">Female</option>
                        <option value="male">Male</option>
                        <option value="other">Other</option>
                      </select>

                      <input
                        type="number"
                        placeholder="Height (cm)"
                        value={formData.height}
                        onChange={(e) => setFormData({ ...formData, height: e.target.value })}
                        className="px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-black focus:border-transparent"
                      />

                      <input
                        type="number"
                        placeholder="Weight (kg)"
                        value={formData.weight}
                        onChange={(e) => setFormData({ ...formData, weight: e.target.value })}
                        className="px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-black focus:border-transparent"
                      />
                    </div>

                    <input
                      type="text"
                      placeholder="Ethnicity (optional)"
                      value={formData.ethnicity}
                      onChange={(e) => setFormData({ ...formData, ethnicity: e.target.value })}
                      className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-black focus:border-transparent"
                    />

                    <button
                      onClick={() => setStep("photo")}
                      className="w-full bg-black text-white py-4 rounded-xl font-semibold hover:bg-gray-800 flex items-center justify-center gap-2"
                    >
                      Continue <ArrowRight className="h-5 w-5" />
                    </button>
                  </div>
                </motion.div>
              )}

              {step === "photo" && (
                <motion.div key="photo" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <div className="space-y-6">
                    <h3 className="text-xl font-semibold text-center">Upload Your Photo</h3>
                    <div className="border-2 border-dashed border-gray-200 rounded-2xl p-8 text-center hover:border-gray-400 transition-colors">
                      <input
                        type="file"
                        accept="image/*"
                        onChange={(e) => {
                          const file = e.target.files?.[0];
                          if (file) handlePhotoUpload(file);
                        }}
                        className="hidden"
                        id="photo-upload"
                      />
                      <label htmlFor="photo-upload" className="cursor-pointer">
                        <div className="text-gray-600">
                          <p className="text-lg font-medium mb-2">Click to upload photo</p>
                          <p className="text-sm">JPG, PNG, WEBP (max 10MB)</p>
                        </div>
                      </label>
                    </div>
                    {loading && (
                      <div className="flex items-center justify-center gap-2 text-gray-600">
                        <Loader2 className="animate-spin h-5 w-5" />
                        <span>Analyzing your photo...</span>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}

              {step === "body_selection" && (
                <motion.div key="body_selection" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <div className="space-y-6">
                    <h3 className="text-xl font-semibold text-center">Choose Your Body Type</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {bodyTypes.map((type) => {
                        const IconComponent = type.icon === "Dumbbell" ? Dumbbell : type.icon === "Users" ? Users : type.icon === "Flame" ? Flame : User;
                        return (
                          <button
                            key={type.id}
                            onClick={() => handleBodyTypeSelect(type.id)}
                            disabled={loading}
                            className="p-6 border-2 border-gray-200 rounded-xl hover:border-black hover:bg-gray-50 transition-all disabled:opacity-50 flex flex-col items-center gap-2"
                          >
                            <IconComponent className="h-8 w-8 text-gray-700" />
                            <span className="font-medium">{type.label}</span>
                          </button>
                        );
                      })}
                    </div>
                    {loading && (
                      <div className="flex items-center justify-center gap-2 text-gray-600">
                        <Loader2 className="animate-spin h-5 w-5" />
                        <span>Generating body options...</span>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}

              {step === "body_generation" && (
                <motion.div key="body_generation" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <div className="space-y-6">
                    <h3 className="text-xl font-semibold text-center">Select Your Preferred Body</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {generatedBodies.map((body) => (
                        <button
                          key={body.id}
                          onClick={() => handleBodySelect(body.data)}
                          disabled={loading}
                          className="aspect-square border-2 border-gray-200 rounded-xl hover:border-black transition-all disabled:opacity-50 overflow-hidden"
                        >
                          <Image src={body.data} alt={`Body ${body.id}`} width={200} height={300} className="object-cover" />
                        </button>
                      ))}
                    </div>
                    {loading && (
                      <div className="flex items-center justify-center gap-2 text-gray-600">
                        <Loader2 className="animate-spin h-5 w-5" />
                        <span>Combining images...</span>
                      </div>
                    )}
                    {composedImage && (
                      <button
                        onClick={handleSubmit}
                        className="w-full bg-black text-white py-4 rounded-xl font-semibold hover:bg-gray-800 flex items-center justify-center gap-2"
                      >
                        Complete Profile <Sparkles className="h-5 w-5" />
                      </button>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {photoType === "full_body" && step === "photo" && (
              <button
                onClick={handleSubmit}
                disabled={loading}
                className="w-full mt-6 bg-black text-white py-4 rounded-xl font-semibold hover:bg-gray-800 disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {loading ? (
                  <>
                    <Loader2 className="animate-spin h-5 w-5" />
                    <span>Creating Profile...</span>
                  </>
                ) : (
                  <>
                    <span>Complete Profile</span>
                    <Sparkles className="h-5 w-5" />
                  </>
                )}
              </button>
            )}
          </div>
        </motion.div>
      </div>
    </div>
  );
}

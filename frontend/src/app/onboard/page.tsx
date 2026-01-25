"use client";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight, Sparkles, Loader2, Dumbbell, Users, Flame, User, Upload, Check, ChevronLeft, Camera } from "lucide-react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";
import { toast } from "sonner";
import { endpoints } from "@/lib/api";
import Image from "next/image";

type Step = "info" | "photo" | "body_selection" | "body_generation" | "complete";

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
  const [generatedBodies, setGeneratedBodies] = useState<Array<{id: string; url: string}>>([]);
  const [selectedBody, setSelectedBody] = useState<string>("");
  const [composedImage, setComposedImage] = useState<string>("");
  const [photoPreview, setPhotoPreview] = useState<string>("");

  const bodyTypes = [
    { id: "athletic", label: "Athletic", icon: Dumbbell, desc: "Toned & fit" },
    { id: "slim", label: "Slim", icon: Users, desc: "Lean build" },
    { id: "muscular", label: "Muscular", icon: Flame, desc: "Strong physique" },
    { id: "average", label: "Average", icon: User, desc: "Medium build" },
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

  const validateInfoStep = () => {
    if (!formData.name.trim()) {
      toast.error("Name is required");
      return false;
    }
    const age = parseInt(formData.age);
    if (isNaN(age) || age < 13 || age > 100) {
      toast.error("Please enter a valid age (13-100)");
      return false;
    }
    const height = parseFloat(formData.height);
    if (isNaN(height) || height < 50 || height < 300) {
      toast.error("Please enter a valid height in cm");
      return false;
    }
    const weight = parseFloat(formData.weight);
    if (isNaN(weight) || weight < 20 || weight > 500) {
      toast.error("Please enter a valid weight in kg");
      return false;
    }
    return true;
  };

  const handlePhotoUpload = async (file: File) => {
    // Validate file size
    if (file.size > 10 * 1024 * 1024) {
      toast.error("File size must be less than 10MB");
      return;
    }

    setFormData({ ...formData, file });
    
    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setPhotoPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
    
    setLoading(true);
    
    try {
      // Analyze if head-only or full-body
      const analysis = await endpoints.analyzeImage(file);
      setPhotoType(analysis.type);
      
      if (analysis.type === "head_only") {
        toast.info("Head-only photo detected! Let's generate a body for you.", { duration: 4000 });
        setTimeout(() => setStep("body_selection"), 1000);
      } else {
        toast.success("Full-body photo detected! You're all set.", { duration: 3000 });
      }
    } catch (error) {
      console.error("Analysis error:", error);
      toast.error("Failed to analyze image. Please try again.");
      setPhotoPreview("");
      setFormData({ ...formData, file: null });
    } finally {
      setLoading(false);
    }
  };

  const handleBodyTypeSelect = async (bodyType: string) => {
    if (!validateInfoStep()) return;
    
    setLoading(true);
    
    try {
      // Generate body options
      const result = await endpoints.generateBodies({
        ethnicity: formData.ethnicity || "Mixed",
        skin_tone: "Medium",
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

  const handleBodySelect = async (bodyUrl: string) => {
    setSelectedBody(bodyUrl);
    setLoading(true);
    
    try {
      if (!formData.file) return;
      
      // Fetch body image from Supabase URL
      const bodyFile = await fetch(bodyUrl).then(r => r.blob()).then(b => new File([b], "body.png"));
      
      // Combine head + body
      const result = await endpoints.combineHeadBody(formData.file, bodyFile);
      setComposedImage(result.image_url);
      
      toast.success("Images combined successfully!");
      // Proceed to save
      await saveProfile(result.image_url);
    } catch (error) {
      console.error("Combination error:", error);
      toast.error("Failed to combine images. Please try again.");
      setLoading(false);
    }
  };

  const saveProfile = async (photoUrl?: string) => {
    const {data: { session } } = await supabase.auth.getSession();
    if (!session) {
      toast.error("Session expired. Please log in again.");
      router.push("/auth");
      return;
    }

    setLoading(true);    
    try {
      let publicUrl = "";
      
      if (photoUrl) {
        // If it's already a Supabase URL from backend, use it directly
        if (photoUrl.includes('supabase.co')) {
          publicUrl = photoUrl;
        } else {
          // If it's a data URL, upload it
          const blob = await fetch(photoUrl).then(r => r.blob());
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
        }
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

      // Create profile - use upsert to handle existing profiles
      const profileData: any = {
        id: session.user.id,
        email: session.user.email,
        name: formData.name.trim(),
        age: parseInt(formData.age),
        gender: formData.gender,
        height_cm: parseFloat(formData.height),
        weight_kg: parseFloat(formData.weight),
        updated_at: new Date().toISOString()
      };

      // Add optional fields
      if (publicUrl) {
        profileData.photo_url = publicUrl;
      }
      if (formData.ethnicity) {
        profileData.ethnicity = formData.ethnicity;
      }

      const { error: profileError } = await supabase
        .from('profiles')
        .upsert(profileData, { onConflict: 'id' });

      if (profileError) {
        console.error("Profile error details:", profileError);
        throw new Error(`Profile creation failed: ${profileError.message}`);
      }
      
      localStorage.setItem("user_name", formData.name.trim());
      setStep("complete");
      toast.success("Profile created successfully!");
      
      setTimeout(() => router.push("/studio"), 2000);
    } catch (error) {
      console.error("Error:", error);
      const message = error instanceof Error ? error.message : "Failed to create profile";
      toast.error(message);
      // Reset to photo step so user can try again
      setStep("photo");
      setPhotoType(null);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async () => {
    if (!validateInfoStep()) return;
    if (!formData.file) {
      toast.error("Please upload a photo");
      return;
    }

    if (photoType === "head_only") {
      if (!selectedBody) {
        toast.error("Please select a body");
        return;
      }
    }

    // Will be saved after body generation flow or directly
    if (photoType === "full_body") {
      await saveProfile();
    } else if (photoType === "head_only" && composedImage) {
      await saveProfile(composedImage);
    }
  };

  const stepProgress = {
    info: 25,
    photo: 50,
    body_selection: 65,
    body_generation: 85,
    complete: 100
  };

  return (
    <div className="min-h-screen pt-20 sm:pt-24 pb-12 px-4 bg-gradient-to-br from-gray-50 via-white to-gray-100">
      <div className="max-w-4xl mx-auto">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
          {/* Header */}
          <div className="text-center mb-6 sm:mb-8">
            <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold text-gray-900 mb-2">Complete Your Profile</h1>
            <p className="text-sm sm:text-base text-gray-600">Let&apos;s get you set up for virtual try-on</p>
          </div>

          {/* Progress Bar */}
          <div className="mb-6 sm:mb-8">
            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-black"
                initial={{ width: 0 }}
                animate={{ width: `${stepProgress[step]}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>
            <p className="text-xs sm:text-sm text-gray-500 mt-2 text-center">{stepProgress[step]}% Complete</p>
          </div>

          {/* Main Card */}
          <div className="bg-white rounded-2xl sm:rounded-3xl p-6 sm:p-8 md:p-10 shadow-xl">
            <AnimatePresence mode="wait">
              {/* Step 1: Info */}
              {step === "info" && (
                <motion.div
                  key="info"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="space-y-5 sm:space-y-6">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Full Name *</label>
                        <input
                          type="text"
                          placeholder="John Doe"
                          value={formData.name}
                          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                          className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-black focus:border-transparent transition-all"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Age *</label>
                        <input
                          type="number"
                          placeholder="25"
                          value={formData.age}
                          onChange={(e) => setFormData({ ...formData, age: e.target.value })}
                          className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-black focus:border-transparent transition-all"
                        />
                      </div>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Gender *</label>
                        <select
                          value={formData.gender}
                          onChange={(e) => setFormData({ ...formData, gender: e.target.value })}
                          className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-black focus:border-transparent transition-all"
                        >
                          <option value="female">Female</option>
                          <option value="male">Male</option>
                          <option value="other">Other</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Height (cm) *</label>
                        <input
                          type="number"
                          placeholder="170"
                          value={formData.height}
                          onChange={(e) => setFormData({ ...formData, height: e.target.value })}
                          className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-black focus:border-transparent transition-all"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Weight (kg) *</label>
                        <input
                          type="number"
                          placeholder="65"
                          value={formData.weight}
                          onChange={(e) => setFormData({ ...formData, weight: e.target.value })}
                          className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-black focus:border-transparent transition-all"
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">Ethnicity (optional)</label>
                      <input
                        type="text"
                        placeholder="e.g., Asian, Caucasian, African, Mixed"
                        value={formData.ethnicity}
                        onChange={(e) => setFormData({ ...formData, ethnicity: e.target.value })}
                        className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-black focus:border-transparent transition-all"
                      />
                    </div>

                    <button
                      onClick={() => {
                        if (validateInfoStep()) setStep("photo");
                      }}
                      className="w-full bg-black text-white py-3 sm:py-4 rounded-xl font-semibold hover:bg-gray-800 transition-all flex items-center justify-center gap-2 touch-manipulation"
                    >
                      Continue <ArrowRight className="h-5 w-5" />
                    </button>
                  </div>
                </motion.div>
              )}

              {/* Step 2: Photo */}
              {step === "photo" && (
                <motion.div
                  key="photo"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="space-y-6">
                    <div className="flex items-center gap-3 mb-4">
                      <button
                        onClick={() => setStep("info")}
                        className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                      >
                        <ChevronLeft className="h-5 w-5" />
                      </button>
                      <h3 className="text-xl sm:text-2xl font-semibold">Upload Your Photo</h3>
                    </div>

                    {!photoPreview ? (
                      <div className="border-2 border-dashed border-gray-300 rounded-2xl p-8 sm:p-12 text-center hover:border-gray-400 transition-colors bg-gray-50">
                        <input
                          type="file"
                          accept="image/*"
                          onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (file) handlePhotoUpload(file);
                          }}
                          className="hidden"
                          id="photo-upload"
                          disabled={loading}
                        />
                        <label htmlFor="photo-upload" className="cursor-pointer">
                          <div className="flex flex-col items-center gap-4">
                            <div className="p-4 bg-black rounded-full">
                              <Camera className="h-8 w-8 text-white" />
                            </div>
                            <div>
                              <p className="text-lg font-semibold text-gray-900 mb-1">Click to upload photo</p>
                              <p className="text-sm text-gray-500">JPG, PNG, WEBP (max 10MB)</p>
                              <p className="text-xs text-gray-400 mt-2">Head-only or full-body photos accepted</p>
                            </div>
                          </div>
                        </label>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <div className="relative rounded-2xl overflow-hidden bg-gray-100 max-w-md mx-auto">
                          <Image
                            src={photoPreview}
                            alt="Preview"
                            width={400}
                            height={600}
                            className="w-full h-auto object-contain"
                          />
                          {photoType && (
                            <div className="absolute top-4 right-4 bg-black/80 text-white px-3 py-1 rounded-full text-sm flex items-center gap-2">
                              <Check className="h-4 w-4" />
                              {photoType === "head_only" ? "Head-only" : "Full-body"}
                            </div>
                          )}
                        </div>
                        
                        {!loading && photoType === "full_body" && (
                          <button
                            onClick={handleSubmit}
                            disabled={loading}
                            className="w-full bg-black text-white py-3 sm:py-4 rounded-xl font-semibold hover:bg-gray-800 disabled:opacity-50 transition-all flex items-center justify-center gap-2"
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
                    )}

                    {loading && (
                      <div className="flex flex-col items-center justify-center gap-3 py-8">
                        <Loader2 className="animate-spin h-8 w-8 text-black" />
                        <p className="text-gray-600 font-medium">Analyzing your photo...</p>
                        <p className="text-sm text-gray-400">This may take up to 60 seconds</p>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}

              {/* Step 3: Body Selection */}
              {step === "body_selection" && (
                <motion.div
                  key="body_selection"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="space-y-6">
                    <div className="flex items-center gap-3 mb-4">
                      <button
                        onClick={() => setStep("photo")}
                        className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                      >
                        <ChevronLeft className="h-5 w-5" />
                      </button>
                      <h3 className="text-xl sm:text-2xl font-semibold">Choose Your Body Type</h3>
                    </div>

                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
                      {bodyTypes.map((type) => {
                        const IconComponent = type.icon;
                        return (
                          <button
                            key={type.id}
                            onClick={() => handleBodyTypeSelect(type.id)}
                            disabled={loading}
                            className="p-4 sm:p-6 border-2 border-gray-200 rounded-xl hover:border-black hover:bg-gray-50 transition-all disabled:opacity-50 flex flex-col items-center gap-2 sm:gap-3 touch-manipulation"
                          >
                            <IconComponent className="h-8 w-8 sm:h-10 sm:w-10 text-gray-700" />
                            <div className="text-center">
                              <p className="font-semibold text-sm sm:text-base">{type.label}</p>
                              <p className="text-xs text-gray-500 mt-1">{type.desc}</p>
                            </div>
                          </button>
                        );
                      })}
                    </div>

                    {loading && (
                      <div className="flex flex-col items-center justify-center gap-3 py-8">
                        <Loader2 className="animate-spin h-8 w-8 text-black" />
                        <p className="text-gray-600 font-medium">Generating body options...</p>
                        <p className="text-sm text-gray-400">This may take a few moments</p>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}

              {/* Step 4: Body Generation */}
              {step === "body_generation" && (
                <motion.div
                  key="body_generation"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="space-y-6">
                    <div className="flex items-center gap-3 mb-4">
                      <button
                        onClick={() => setStep("body_selection")}
                        className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                        disabled={loading}
                      >
                        <ChevronLeft className="h-5 w-5" />
                      </button>
                      <h3 className="text-xl sm:text-2xl font-semibold">Select Your Preferred Body</h3>
                    </div>

                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
                      {generatedBodies.map((body) => (
                        <button
                          key={body.id}
                          onClick={() => handleBodySelect(body.url)}
                          disabled={loading}
                          className="aspect-[3/4] border-2 border-gray-200 rounded-xl hover:border-black transition-all disabled:opacity-50 overflow-hidden relative group"
                        >
                          <Image
                            src={body.url}
                            alt={`Body ${body.id}`}
                            fill
                            className="object-cover group-hover:scale-105 transition-transform duration-300"
                          />
                          {selectedBody === body.url && (
                            <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                              <Check className="h-12 w-12 text-white" />
                            </div>
                          )}
                        </button>
                      ))}
                    </div>

                    {loading && (
                      <div className="flex flex-col items-center justify-center gap-3 py-8">
                        <Loader2 className="animate-spin h-8 w-8 text-black" />
                        <p className="text-gray-600 font-medium">Combining images...</p>
                        <p className="text-sm text-gray-400">Creating your perfect profile photo</p>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}

              {/* Step 5: Complete */}
              {step === "complete" && (
                <motion.div
                  key="complete"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5 }}
                >
                  <div className="text-center py-8 sm:py-12">
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
                      className="inline-flex items-center justify-center w-16 h-16 sm:w-20 sm:h-20 bg-green-100 rounded-full mb-6"
                    >
                      <Check className="h-8 w-8 sm:h-10 sm:w-10 text-green-600" />
                    </motion.div>
                    <h2 className="text-2xl sm:text-3xl font-bold text-gray-900 mb-3">Profile Complete!</h2>
                    <p className="text-gray-600 mb-6">Redirecting you to the studio...</p>
                    <div className="flex items-center justify-center gap-2">
                      <Loader2 className="animate-spin h-5 w-5 text-gray-400" />
                      <span className="text-sm text-gray-500">Please wait</span>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

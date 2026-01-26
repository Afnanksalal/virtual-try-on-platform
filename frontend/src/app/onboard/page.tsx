"use client";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight, Sparkles, Loader2, Dumbbell, Users, Flame, User, Upload, Check, ChevronLeft, Camera } from "lucide-react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";
import { toast } from "sonner";
import { endpoints } from "@/lib/api";
import Image from "next/image";

type Step = "welcome" | "upload" | "analyzing" | "parameters" | "generating" | "complete";

interface BodyParameters {
  ethnicity: string;
  bodyType: string;
  height: number;
  weight: number;
  gender: string;
  age: number;
  name: string;
}

interface ImageAnalysis {
  type: "head_only" | "full_body";
  confidence: number;
}

interface OnboardingState {
  currentStep: Step;
  uploadedImage: File | null;
  imageAnalysis: ImageAnalysis | null;
  bodyParameters: BodyParameters | null;
  generatedBodyUrl: string | null;
  canGoBack: boolean;
  canGoForward: boolean;
}

export default function OnboardPage() {
  const router = useRouter();
  
  // State management
  const [currentStep, setCurrentStep] = useState<Step>("welcome");
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [imageAnalysis, setImageAnalysis] = useState<ImageAnalysis | null>(null);
  const [bodyParameters, setBodyParameters] = useState<BodyParameters>({
    name: "",
    age: 0,
    gender: "female",
    height: 0,
    weight: 0,
    ethnicity: "",
    bodyType: ""
  });
  const [generatedBodyUrl, setGeneratedBodyUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [photoPreview, setPhotoPreview] = useState<string>("");
  const [generatedBodies, setGeneratedBodies] = useState<Array<{id: string; url: string}>>([]);
  const [selectedBodyType, setSelectedBodyType] = useState<string>("");
  const [composedImage, setComposedImage] = useState<string>("");

  const bodyTypes = [
    { id: "athletic", label: "Athletic", icon: Dumbbell, desc: "Toned & fit" },
    { id: "slim", label: "Slim", icon: Users, desc: "Lean build" },
    { id: "muscular", label: "Muscular", icon: Flame, desc: "Strong physique" },
    { id: "average", label: "Average", icon: User, desc: "Medium build" },
  ];

  // Navigation helpers
  const canGoBack = currentStep !== "welcome" && currentStep !== "analyzing" && currentStep !== "generating" && currentStep !== "complete";
  const canGoForward = false; // Determined by step validation

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

  // Load progress from localStorage
  useEffect(() => {
    const savedProgress = localStorage.getItem("onboarding_progress");
    if (savedProgress) {
      try {
        const progress = JSON.parse(savedProgress);
        if (progress.currentStep && progress.currentStep !== "complete") {
          setCurrentStep(progress.currentStep);
          if (progress.bodyParameters) {
            setBodyParameters(progress.bodyParameters);
          }
          if (progress.imageAnalysis) {
            setImageAnalysis(progress.imageAnalysis);
          }
          toast.info("Resuming your onboarding progress");
        }
      } catch (error) {
        console.error("Failed to load progress:", error);
      }
    }
  }, []);

  // Save progress to localStorage
  useEffect(() => {
    if (currentStep !== "welcome" && currentStep !== "complete") {
      const progress = {
        currentStep,
        bodyParameters,
        imageAnalysis,
        timestamp: Date.now()
      };
      localStorage.setItem("onboarding_progress", JSON.stringify(progress));
    }
  }, [currentStep, bodyParameters, imageAnalysis]);

  // Step navigation
  const handleNext = () => {
    const stepOrder: Step[] = ["welcome", "upload", "analyzing", "parameters", "generating", "complete"];
    const currentIndex = stepOrder.indexOf(currentStep);
    if (currentIndex < stepOrder.length - 1) {
      setCurrentStep(stepOrder[currentIndex + 1]);
    }
  };

  const handleBack = () => {
    if (!canGoBack) return;
    
    const stepOrder: Step[] = ["welcome", "upload", "analyzing", "parameters", "generating", "complete"];
    const currentIndex = stepOrder.indexOf(currentStep);
    
    // Skip analyzing and generating steps when going back
    if (currentIndex > 0) {
      let previousStep = stepOrder[currentIndex - 1];
      if (previousStep === "analyzing" || previousStep === "generating") {
        previousStep = stepOrder[currentIndex - 2];
      }
      setCurrentStep(previousStep);
    }
  };

  // Validation
  const validateParameters = () => {
    if (!bodyParameters.name.trim()) {
      toast.error("Name is required");
      return false;
    }
    if (bodyParameters.age < 13 || bodyParameters.age > 100) {
      toast.error("Please enter a valid age (13-100)");
      return false;
    }
    if (bodyParameters.height < 50 || bodyParameters.height > 300) {
      toast.error("Please enter a valid height in cm");
      return false;
    }
    if (bodyParameters.weight < 20 || bodyParameters.weight > 500) {
      toast.error("Please enter a valid weight in kg");
      return false;
    }
    // Body type only required for head-only photos
    if (imageAnalysis?.type === "head_only" && !bodyParameters.bodyType) {
      toast.error("Please select a body type");
      return false;
    }
    return true;
  };

  // Image upload and analysis
  const handlePhotoUpload = async (file: File) => {
    // Validate file size
    if (file.size > 10 * 1024 * 1024) {
      toast.error("File size must be less than 10MB");
      return;
    }

    setUploadedImage(file);
    
    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setPhotoPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
    
    // Move to analyzing step
    setCurrentStep("analyzing");
    setLoading(true);
    
    try {
      // Analyze if head-only or full-body
      const analysis = await endpoints.analyzeImage(file);
      setImageAnalysis({
        type: analysis.type,
        confidence: analysis.confidence || 0.9
      });
      
      if (analysis.type === "head_only") {
        toast.info("Head-only photo detected! Let's set up your body parameters.", { duration: 4000 });
        setCurrentStep("parameters");
      } else {
        toast.success("Full-body photo detected! Let's collect some basic info.", { duration: 3000 });
        // Still need basic parameters even for full-body images
        setCurrentStep("parameters");
      }
    } catch (error) {
      console.error("Analysis error:", error);
      toast.error("Failed to analyze image. Please try again.");
      setPhotoPreview("");
      setUploadedImage(null);
      setCurrentStep("upload");
    } finally {
      setLoading(false);
    }
  };

  // Body generation or profile save
  const handleParametersSubmit = async () => {
    if (!validateParameters()) return;
    
    // If full-body photo, skip body generation and save directly
    if (imageAnalysis?.type === "full_body") {
      await saveProfile();
      return;
    }
    
    // If head-only photo, generate body options
    setCurrentStep("generating");
    setLoading(true);
    
    try {
      // Generate body options
      const result = await endpoints.generateBodies({
        ethnicity: bodyParameters.ethnicity || "Mixed",
        skin_tone: "Medium",
        body_type: bodyParameters.bodyType,
        height_cm: bodyParameters.height,
        weight_kg: bodyParameters.weight,
      });
      
      setGeneratedBodies(result.images);
      toast.success("Body options generated!");
    } catch (error) {
      console.error("Body generation error:", error);
      toast.error("Failed to generate bodies. Please try again.");
      setCurrentStep("parameters");
    } finally {
      setLoading(false);
    }
  };

  const handleBodySelect = async (bodyUrl: string) => {
    setGeneratedBodyUrl(bodyUrl);
    setLoading(true);
    
    try {
      if (!uploadedImage) return;
      
      // Fetch body image from Supabase URL
      const bodyFile = await fetch(bodyUrl).then(r => r.blob()).then(b => new File([b], "body.png"));
      
      // Combine head + body
      const result = await endpoints.combineHeadBody(uploadedImage, bodyFile);
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

  // Save profile
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
        if (!uploadedImage) throw new Error("No photo");
        
        const fileName = `${session.user.id}/${Date.now()}_${uploadedImage.name}`;
        const { error: uploadError } = await supabase.storage
          .from('user-uploads')
          .upload(fileName, uploadedImage, { cacheControl: '3600', upsert: false });

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
        name: bodyParameters.name.trim(),
        age: bodyParameters.age,
        gender: bodyParameters.gender,
        height_cm: bodyParameters.height,
        weight_kg: bodyParameters.weight,
        updated_at: new Date().toISOString()
      };

      // Add optional fields
      if (publicUrl) {
        profileData.photo_url = publicUrl;
      }
      if (bodyParameters.ethnicity) {
        profileData.ethnicity = bodyParameters.ethnicity;
      }

      const { error: profileError } = await supabase
        .from('profiles')
        .upsert(profileData, { onConflict: 'id' });

      if (profileError) {
        console.error("Profile error details:", profileError);
        throw new Error(`Profile creation failed: ${profileError.message}`);
      }
      
      localStorage.setItem("user_name", bodyParameters.name.trim());
      localStorage.removeItem("onboarding_progress"); // Clear progress
      setCurrentStep("complete");
      toast.success("Profile created successfully!");
      
      setTimeout(() => router.push("/studio"), 2000);
    } catch (error) {
      console.error("Error:", error);
      const message = error instanceof Error ? error.message : "Failed to create profile";
      toast.error(message);
      // Reset to upload step so user can try again
      setCurrentStep("upload");
      setImageAnalysis(null);
    } finally {
      setLoading(false);
    }
  };

  const handleComplete = () => {
    localStorage.removeItem("onboarding_progress");
    router.push("/studio");
  };

  const stepProgress = {
    welcome: 0,
    upload: 25,
    analyzing: 40,
    parameters: 60,
    generating: 80,
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
                animate={{ width: `${stepProgress[currentStep]}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>
            <p className="text-xs sm:text-sm text-gray-500 mt-2 text-center">{stepProgress[currentStep]}% Complete</p>
          </div>

          {/* Main Card */}
          <div className="bg-white rounded-2xl sm:rounded-3xl p-6 sm:p-8 md:p-10 shadow-xl">
            <AnimatePresence mode="wait">
              {/* Welcome Step */}
              {currentStep === "welcome" && (
                <motion.div
                  key="welcome"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="text-center space-y-6 py-8">
                    <div className="inline-flex items-center justify-center w-20 h-20 bg-black rounded-full mb-4">
                      <Sparkles className="h-10 w-10 text-white" />
                    </div>
                    <h2 className="text-2xl sm:text-3xl font-bold text-gray-900">Welcome to Virtual Try-On</h2>
                    <p className="text-gray-600 max-w-2xl mx-auto">
                      We&apos;ll guide you through a quick setup process to create your profile. 
                      This will help us provide the best virtual try-on experience tailored to you.
                    </p>
                    <div className="space-y-3 text-left max-w-md mx-auto mt-8">
                      <div className="flex items-start gap-3">
                        <div className="flex-shrink-0 w-6 h-6 bg-black rounded-full flex items-center justify-center text-white text-sm font-bold">1</div>
                        <div>
                          <p className="font-semibold text-gray-900">Upload Your Photo</p>
                          <p className="text-sm text-gray-600">Head-only or full-body photos work great</p>
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <div className="flex-shrink-0 w-6 h-6 bg-black rounded-full flex items-center justify-center text-white text-sm font-bold">2</div>
                        <div>
                          <p className="font-semibold text-gray-900">Set Your Preferences</p>
                          <p className="text-sm text-gray-600">Tell us about your body type and style</p>
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <div className="flex-shrink-0 w-6 h-6 bg-black rounded-full flex items-center justify-center text-white text-sm font-bold">3</div>
                        <div>
                          <p className="font-semibold text-gray-900">Start Trying On</p>
                          <p className="text-sm text-gray-600">Explore outfits and see yourself in new styles</p>
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => setCurrentStep("upload")}
                      className="mt-8 bg-black text-white px-8 py-4 rounded-xl font-semibold hover:bg-gray-800 transition-all flex items-center justify-center gap-2 mx-auto touch-manipulation"
                    >
                      Get Started <ArrowRight className="h-5 w-5" />
                    </button>
                  </div>
                </motion.div>
              )}

              {/* Upload Step */}
              {currentStep === "upload" && (
                <motion.div
                  key="upload"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="space-y-6">
                    <div className="flex items-center gap-3 mb-4">
                      {canGoBack && (
                        <button
                          onClick={handleBack}
                          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                        >
                          <ChevronLeft className="h-5 w-5" />
                        </button>
                      )}
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
                          {imageAnalysis && (
                            <div className="absolute top-4 right-4 bg-black/80 text-white px-3 py-1 rounded-full text-sm flex items-center gap-2">
                              <Check className="h-4 w-4" />
                              {imageAnalysis.type === "head_only" ? "Head-only" : "Full-body"}
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}

              {/* Analyzing Step */}
              {currentStep === "analyzing" && (
                <motion.div
                  key="analyzing"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="flex flex-col items-center justify-center gap-4 py-12">
                    <Loader2 className="animate-spin h-12 w-12 text-black" />
                    <h3 className="text-xl font-semibold text-gray-900">Analyzing Your Photo</h3>
                    <p className="text-gray-600 text-center max-w-md">
                      We&apos;re detecting whether your photo is head-only or full-body to provide the best experience.
                    </p>
                    <p className="text-sm text-gray-400">This may take up to 60 seconds</p>
                  </div>
                </motion.div>
              )}

              {/* Parameters Step */}
              {currentStep === "parameters" && (
                <motion.div
                  key="parameters"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="space-y-6">
                    <div className="flex items-center gap-3 mb-4">
                      {canGoBack && (
                        <button
                          onClick={handleBack}
                          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                        >
                          <ChevronLeft className="h-5 w-5" />
                        </button>
                      )}
                      <h3 className="text-xl sm:text-2xl font-semibold">Set Your Body Parameters</h3>
                    </div>

                    {imageAnalysis && (
                      <div className={`border rounded-xl p-4 mb-6 ${
                        imageAnalysis.type === "head_only" 
                          ? "bg-blue-50 border-blue-200" 
                          : "bg-green-50 border-green-200"
                      }`}>
                        <p className={`text-sm ${
                          imageAnalysis.type === "head_only" 
                            ? "text-blue-900" 
                            : "text-green-900"
                        }`}>
                          {imageAnalysis.type === "head_only" ? (
                            <>
                              <strong>Head-only photo detected!</strong> We&apos;ll generate a body model based on your preferences.
                            </>
                          ) : (
                            <>
                              <strong>Full-body photo detected!</strong> We&apos;ll use your photo directly for try-ons.
                            </>
                          )}
                        </p>
                      </div>
                    )}

                    <div className="space-y-5">
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">Full Name *</label>
                          <input
                            type="text"
                            placeholder="John Doe"
                            value={bodyParameters.name}
                            onChange={(e) => setBodyParameters({ 
                              ...bodyParameters, 
                              name: e.target.value 
                            })}
                            className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-black focus:border-transparent transition-all"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">Age *</label>
                          <input
                            type="number"
                            placeholder="25"
                            value={bodyParameters.age || ""}
                            onChange={(e) => setBodyParameters({ 
                              ...bodyParameters, 
                              age: parseInt(e.target.value) || 0 
                            })}
                            className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-black focus:border-transparent transition-all"
                          />
                        </div>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">Gender *</label>
                          <select
                            value={bodyParameters.gender}
                            onChange={(e) => setBodyParameters({ 
                              ...bodyParameters, 
                              gender: e.target.value 
                            })}
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
                            value={bodyParameters.height || ""}
                            onChange={(e) => setBodyParameters({ 
                              ...bodyParameters, 
                              height: parseFloat(e.target.value) || 0 
                            })}
                            className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-black focus:border-transparent transition-all"
                          />
                        </div>

                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-2">Weight (kg) *</label>
                          <input
                            type="number"
                            placeholder="65"
                            value={bodyParameters.weight || ""}
                            onChange={(e) => setBodyParameters({ 
                              ...bodyParameters, 
                              weight: parseFloat(e.target.value) || 0 
                            })}
                            className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-black focus:border-transparent transition-all"
                          />
                        </div>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">Ethnicity (optional)</label>
                        <input
                          type="text"
                          placeholder="e.g., Asian, Caucasian, African, Mixed"
                          value={bodyParameters.ethnicity}
                          onChange={(e) => setBodyParameters({ 
                            ...bodyParameters, 
                            ethnicity: e.target.value 
                          })}
                          className="w-full px-4 py-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-black focus:border-transparent transition-all"
                        />
                      </div>

                      {/* Body Type - Only for head-only photos */}
                      {imageAnalysis?.type === "head_only" && (
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-3">Body Type *</label>
                          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                            {bodyTypes.map((type) => {
                              const IconComponent = type.icon;
                              const isSelected = bodyParameters.bodyType === type.id;
                              return (
                                <button
                                  key={type.id}
                                  onClick={() => setBodyParameters({ 
                                    ...bodyParameters, 
                                    bodyType: type.id 
                                  })}
                                  className={`p-4 border-2 rounded-xl transition-all flex flex-col items-center gap-2 ${
                                    isSelected 
                                      ? 'border-black bg-gray-50' 
                                      : 'border-gray-200 hover:border-gray-400'
                                  }`}
                                >
                                  <IconComponent className="h-8 w-8 text-gray-700" />
                                  <div className="text-center">
                                    <p className="font-semibold text-sm">{type.label}</p>
                                    <p className="text-xs text-gray-500">{type.desc}</p>
                                  </div>
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      <button
                        onClick={handleParametersSubmit}
                        disabled={loading || (imageAnalysis?.type === "head_only" && !bodyParameters.bodyType)}
                        className="w-full bg-black text-white py-3 sm:py-4 rounded-xl font-semibold hover:bg-gray-800 disabled:opacity-50 transition-all flex items-center justify-center gap-2"
                      >
                        Generate Body Options <ArrowRight className="h-5 w-5" />
                      </button>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Generating Step */}
              {currentStep === "generating" && (
                <motion.div
                  key="generating"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="space-y-6">
                    {generatedBodies.length === 0 ? (
                      <div className="flex flex-col items-center justify-center gap-4 py-12">
                        <Loader2 className="animate-spin h-12 w-12 text-black" />
                        <h3 className="text-xl font-semibold text-gray-900">Generating Body Models</h3>
                        <p className="text-gray-600 text-center max-w-md">
                          Creating personalized body options based on your parameters. This may take a few moments.
                        </p>
                        <div className="w-full max-w-xs bg-gray-200 rounded-full h-2 mt-4">
                          <motion.div
                            className="bg-black h-2 rounded-full"
                            initial={{ width: "0%" }}
                            animate={{ width: "100%" }}
                            transition={{ duration: 10, ease: "linear" }}
                          />
                        </div>
                      </div>
                    ) : (
                      <>
                        <div className="flex items-center gap-3 mb-4">
                          {canGoBack && !loading && (
                            <button
                              onClick={handleBack}
                              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                            >
                              <ChevronLeft className="h-5 w-5" />
                            </button>
                          )}
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
                              {generatedBodyUrl === body.url && (
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
                      </>
                    )}
                  </div>
                </motion.div>
              )}

              {/* Complete Step */}
              {currentStep === "complete" && (
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

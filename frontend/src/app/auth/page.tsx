"use client";
import { useState } from "react";
import { supabase } from "@/lib/supabase";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Lock, Mail, ArrowRight, Loader2 } from "lucide-react";
import { toast } from "sonner";

export default function AuthPage() {
  const router = useRouter();
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    
    // Strict Validation
    if (!formData.email.includes("@")) {
        toast.error("Please enter a valid email address");
        setLoading(false);
        return;
    }
    if (formData.password.length < 8) {
        toast.error("Password must be at least 8 characters");
        setLoading(false);
        return;
    }

    try {
      let authUser = null;

      if (isLogin) {
        const { data, error } = await supabase.auth.signInWithPassword({
          email: formData.email,
          password: formData.password,
        });
        if (error) throw error;
        authUser = data.user;
      } else {
        const { data, error } = await supabase.auth.signUp({
          email: formData.email,
          password: formData.password,
        });
        if (error) throw error;
        authUser = data.user;
        if (!isLogin) {
          toast.success("Account created successfully! Complete your profile to continue.");
        }
      }

      if (authUser) {
         // Store ID for easy access (optional, since we have session)
         localStorage.setItem("vton_user_id", authUser.id);

         // Check if profile exists
         // If table doesn't exist or profile not found, go to onboard
         try {
           const { data: profile, error } = await supabase
             .from("profiles")
             .select("id")
             .eq("id", authUser.id)
             .maybeSingle(); // Use maybeSingle to avoid error if no rows found
           
           if (error) {
             console.warn("Profile check error (table may not exist):", error);
             // If table doesn't exist, user needs to onboard
             router.push("/onboard");
           } else if (profile) {
             // Profile exists, go to studio
             router.push("/studio");
           } else {
             // No profile found, go to onboard
             router.push("/onboard");
           }
         } catch (err) {
           console.warn("Profile check failed:", err);
           // On any error, default to onboard
           router.push("/onboard");
         }
       }

    } catch (error) {
      const message = error instanceof Error ? error.message : "Authentication failed";
      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4 sm:px-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md"
      >
        <div className="bg-white rounded-3xl shadow-2xl p-6 sm:p-8 md:p-10">
          <div className="text-center mb-6 sm:mb-8">
            <h1 className="text-2xl sm:text-3xl font-bold text-gray-900 mb-2">
              {isLogin ? "Welcome Back" : "Get Started"}
            </h1>
            <p className="text-sm sm:text-base text-gray-500">
              {isLogin ? "Sign in to continue" : "Create your account"}
            </p>
          </div>

          <div className="flex bg-gray-100 rounded-xl p-1 mb-8">
            <motion.button 
              whileTap={{ scale: 0.95 }}
              onClick={() => setIsLogin(true)}
              className={`flex-1 py-3 rounded-lg font-medium transition-all ${
                isLogin ? "bg-white shadow-sm text-gray-900" : "text-gray-500"
              }`}
            >
              Sign In
            </motion.button>
            <motion.button 
              whileTap={{ scale: 0.95 }}
              onClick={() => setIsLogin(false)}
              className={`flex-1 py-3 rounded-lg font-medium transition-all ${
                !isLogin ? "bg-white shadow-sm text-gray-900" : "text-gray-500"
              }`}
            >
              Sign Up
            </motion.button>
          </div>

          <form onSubmit={handleAuth} className="space-y-4">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
            >
              <div className="relative">
                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
                <input
                  type="email"
                  placeholder="Email address"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  className="w-full pl-12 pr-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent transition-all"
                  required
                />
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <div className="relative">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
                <input
                  type="password"
                  placeholder="Password"
                  value={formData.password}
                  onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                  className="w-full pl-12 pr-4 py-3 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-black focus:border-transparent transition-all"
                  required
                />
              </div>
            </motion.div>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              type="submit"
              disabled={loading}
              className="w-full bg-black text-white py-3 sm:py-4 rounded-xl font-semibold hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 touch-manipulation transition-colors"
            >
              {loading ? (
                <>
                  <Loader2 className="animate-spin h-5 w-5" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <span>{isLogin ? "Sign In" : "Create Account"}</span>
                  <ArrowRight className="h-5 w-5" />
                </>
              )}
            </motion.button>
          </form>
          
          <div className="mt-6 text-center">
            <button
              type="button"
              onClick={() => setIsLogin(!isLogin)}
              className="text-sm sm:text-base text-gray-600 hover:text-gray-900 font-medium transition-colors touch-manipulation"
            >
              {isLogin ? "Don't have an account? " : "Already have an account? "}
              <span className="text-black hover:underline font-semibold">
                {isLogin ? "Sign up" : "Sign in"}
              </span>
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}

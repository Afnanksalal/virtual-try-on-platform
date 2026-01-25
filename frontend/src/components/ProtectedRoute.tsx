"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { supabase } from "@/lib/supabase";

export default function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [authorized, setAuthorized] = useState(false);

  useEffect(() => {
    const checkAuth = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      
      if (!session) {
        // Double check local storage as fallback but prefer session
        const localId = localStorage.getItem("vton_user_id");
        if (!localId) {
             router.replace("/auth");
             return; // Stop here - don't set authorized
        }
        // Has local ID but no session - allow but should re-auth soon
        setAuthorized(true);
        return; // Stop here
      }
      // Has valid session
      setAuthorized(true);
    };
    checkAuth();
  }, [router]);

  if (!authorized) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#FAF9F6]">
        <Loader2 className="h-8 w-8 animate-spin text-stone-400" />
      </div>
    );
  }

  return <>{children}</>;
}

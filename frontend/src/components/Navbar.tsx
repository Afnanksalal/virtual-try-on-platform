"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { User, LogOut, Menu, X } from "lucide-react";
import { supabase } from "@/lib/supabase";
import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";

export default function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const [user, setUser] = useState<{ email?: string } | null>(null);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setUser(session?.user ?? null);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
    });

    return () => subscription.unsubscribe();
  }, []);

  const handleLogout = async () => {
    await supabase.auth.signOut();
    router.push("/");
  };

  const navLinks = [
    { href: "/studio", label: "Studio" },
    { href: "/wardrobe", label: "Wardrobe" },
    { href: "/shop", label: "Shop" },
  ];

  return (
    <>
      <nav className="fixed top-4 left-1/2 -translate-x-1/2 w-full max-w-6xl px-4 z-50">
        <div className="bg-white/80 backdrop-blur-lg rounded-2xl shadow-lg border border-gray-200/50 px-6 py-3">
          <div className="flex items-center justify-between">
            {/* Logo */}
            <Link href="/" className="text-xl font-bold text-gray-900 hover:text-gray-700 transition-colors">
              VirtualFit
            </Link>

            {/* Desktop Nav Links */}
            <div className="hidden md:flex items-center gap-1">
              {navLinks.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                    pathname === link.href
                      ? "bg-black text-white"
                      : "text-gray-700 hover:bg-gray-100"
                  }`}
                >
                  {link.label}
                </Link>
              ))}
            </div>

            {/* Desktop User Section */}
            <div className="hidden md:flex items-center gap-2">
              {user ? (
                <>
                  <span className="text-sm text-gray-600">
                    {user.email?.split('@')[0]}
                  </span>
                  <button
                    onClick={handleLogout}
                    className="flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-medium text-red-600 hover:bg-red-50 transition-colors cursor-pointer"
                  >
                    <LogOut className="h-4 w-4" />
                    Logout
                  </button>
                </>
              ) : (
                <Link
                  href="/auth"
                  className="flex items-center gap-2 px-4 py-2 rounded-xl bg-black text-white text-sm font-medium hover:bg-gray-800 transition-colors"
                >
                  <User className="h-4 w-4" />
                  Login
                </Link>
              )}
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 rounded-lg hover:bg-gray-100 transition-colors"
            >
              {mobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>
        </div>
      </nav>

      {/* Full Screen Mobile Menu */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 bg-white z-40 md:hidden">
          <div className="h-full flex flex-col">
            {/* Header Spacer */}
            <div className="h-20"></div>
            
            {/* Menu Content */}
            <div className="flex-1 overflow-y-auto px-6 py-6">
              <div className="space-y-2">
                {navLinks.map((link) => (
                  <Link
                    key={link.href}
                    href={link.href}
                    onClick={() => setMobileMenuOpen(false)}
                    className={`block px-6 py-4 rounded-xl text-base font-medium transition-all ${
                      pathname === link.href
                        ? "bg-black text-white"
                        : "text-gray-700 hover:bg-gray-100"
                    }`}
                  >
                    {link.label}
                  </Link>
                ))}
                
                {user ? (
                  <div className="pt-8 mt-8 border-t border-gray-200 space-y-4">
                    <div className="px-6 py-4 bg-gray-50 rounded-xl">
                      <p className="text-xs text-gray-500 mb-1 uppercase tracking-wide">Signed in as</p>
                      <p className="text-base font-semibold text-gray-900">{user.email?.split('@')[0]}</p>
                    </div>
                    <button
                      onClick={() => {
                        handleLogout();
                        setMobileMenuOpen(false);
                      }}
                      className="w-full flex items-center justify-center gap-2 px-6 py-4 rounded-xl text-base font-semibold text-red-600 bg-red-50 hover:bg-red-100 transition-colors"
                    >
                      <LogOut className="h-5 w-5" />
                      Logout
                    </button>
                  </div>
                ) : (
                  <Link
                    href="/auth"
                    onClick={() => setMobileMenuOpen(false)}
                    className="block text-center px-6 py-4 rounded-xl bg-black text-white text-base font-semibold hover:bg-gray-800 transition-colors mt-8"
                  >
                    Login
                  </Link>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

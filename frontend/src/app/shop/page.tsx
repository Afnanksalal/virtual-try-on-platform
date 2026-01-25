"use client";
import { useState } from "react";
import Image from "next/image";
import { ShoppingBag } from "lucide-react";

export default function ShopPage() {
  // Mock data - will be replaced with Supabase product catalog
  const [items] = useState([
    {
      id: "1",
      name: "Classic Black Dress",
      image_url: "https://placehold.co/400x600/000000/FFFFFF/png?text=Black+Dress",
      price: 89.99,
      category: "Dresses",
    },
    {
      id: "2",
      name: "Casual Denim Jacket",
      image_url: "https://placehold.co/400x600/4169E1/FFFFFF/png?text=Denim+Jacket",
      price: 129.99,
      category: "Outerwear",
    },
    {
      id: "3",
      name: "Summer Floral Top",
      image_url: "https://placehold.co/400x600/FF1493/FFFFFF/png?text=Floral+Top",
      price: 49.99,
      category: "Tops",
    },
    {
      id: "4",
      name: "Slim Fit Jeans",
      image_url: "https://placehold.co/400x600/191970/FFFFFF/png?text=Jeans",
      price: 79.99,
      category: "Bottoms",
    },
  ]);

  return (
    <div className="min-h-screen pt-28 pb-12 px-4 bg-gray-50">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6 sm:mb-8">
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-gray-900 mb-2">Curated Shop</h1>
          <p className="text-sm sm:text-base text-gray-600">Discover items selected for your style</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {items.map((item) => (
            <div key={item.id} className="bg-white rounded-2xl overflow-hidden shadow-sm hover:shadow-md transition-shadow">
              <div className="relative aspect-3/4">
                <Image 
                  src={item.image_url} 
                  alt={item.name} 
                  fill 
                  sizes="(max-width: 768px) 100vw, (max-width: 1024px) 50vw, 25vw"
                  priority={item.id === "1"}
                  className="object-cover"
                />
              </div>
              <div className="p-4">
                <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">{item.category}</p>
                <h3 className="font-semibold text-gray-900 mb-2">{item.name}</h3>
                <div className="flex items-center justify-between">
                  <p className="text-lg font-bold text-gray-900">${item.price}</p>
                  <button className="bg-black text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-gray-800 transition-colors flex items-center gap-2">
                    <ShoppingBag className="w-4 h-4" />
                    Buy
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

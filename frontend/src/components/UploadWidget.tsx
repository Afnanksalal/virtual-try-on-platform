"use client";
import { Upload, X, Image as ImageIcon } from "lucide-react";
import { useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Image from "next/image";

interface UploadWidgetProps {
    onFileSelect: (file: File) => void;
}

export default function UploadWidget({ onFileSelect }: UploadWidgetProps) {
    const [preview, setPreview] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setPreview(URL.createObjectURL(file));
            onFileSelect(file);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        const file = e.dataTransfer.files?.[0];
        if (file) {
            setPreview(URL.createObjectURL(file));
            onFileSelect(file);
        }
    };

    const clearFile = () => {
        setPreview(null);
        if (fileInputRef.current) fileInputRef.current.value = "";
    };

    return (
        <div className="w-full">
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                className="hidden"
                accept="image/*"
            />

            <AnimatePresence mode="wait">
                {!preview ? (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}
                        onClick={() => fileInputRef.current?.click()}
                        className="border-2 border-dashed border-gray-300 rounded-2xl p-12 text-center cursor-pointer hover:border-primary-500 hover:bg-primary-50/50 transition-colors group"
                    >
                        <div className="h-16 w-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform">
                            <Upload className="h-8 w-8 text-gray-400 group-hover:text-primary-600" />
                        </div>
                        <h3 className="text-lg font-semibold text-gray-900">Upload your photo</h3>
                        <p className="text-gray-500 mt-2 text-sm">Drag & drop or click to browse</p>
                        <p className="text-xs text-gray-400 mt-4">JPG, PNG up to 10MB</p>
                    </motion.div>
                ) : (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        className="relative rounded-2xl overflow-hidden shadow-lg h-96 w-full"
                    >
                        <Image src={preview} alt="Preview" fill className="object-cover" />
                        <button
                            onClick={clearFile}
                            className="absolute top-4 right-4 bg-white/80 p-2 rounded-full hover:bg-white text-gray-800 transition-colors backdrop-blur-sm shadow-sm z-10"
                        >
                            <X className="h-5 w-5" />
                        </button>
                        <div className="absolute bottom-0 left-0 right-0 p-4 bg-linear-to-t from-black/50 to-transparent">
                            <div className="flex items-center gap-2 text-white">
                                <ImageIcon className="h-4 w-4" />
                                <span className="text-sm font-medium">Photo uploaded</span>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

"use client";
import { Canvas } from "@react-three/fiber";
import { useGLTF, Stage, OrbitControls, Grid } from "@react-three/drei";
import { Suspense, useState } from "react";
import { Download, Grid3X3, Loader2, AlertCircle } from "lucide-react";
import { toast } from "sonner";
import { endpoints } from "@/lib/api";

interface ModelViewerProps {
  modelUrl: string;
  format?: 'glb' | 'obj' | 'ply';
  downloadToken?: string;
  onDownload?: () => void;
}

function Model({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  return <primitive object={scene} />;
}

function SceneContent({ modelUrl, showGrid }: { modelUrl: string; showGrid: boolean }) {
  return (
    <>
      {/* Lighting Setup */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[10, 10, 5]} intensity={1} castShadow />
      <pointLight position={[-10, -10, -5]} intensity={0.3} />
      <spotLight position={[0, 10, 0]} angle={0.3} penumbra={1} intensity={0.5} castShadow />

      {/* Model */}
      <Suspense fallback={null}>
        <Stage environment="city" intensity={0.6}>
          <Model url={modelUrl} />
        </Stage>
      </Suspense>

      {/* Grid Helper */}
      {showGrid && <Grid infiniteGrid fadeDistance={50} fadeStrength={2} />}

      {/* Orbit Controls */}
      <OrbitControls
        enableZoom={true}
        enablePan={true}
        enableRotate={true}
        minDistance={2}
        maxDistance={10}
        dampingFactor={0.05}
        enableDamping
      />
    </>
  );
}

export default function ModelViewer({ 
  modelUrl, 
  format = 'glb',
  downloadToken,
  onDownload 
}: ModelViewerProps) {
  const [showGrid, setShowGrid] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);

  const handleDownload = async () => {
    if (isDownloading) return;

    try {
      setIsDownloading(true);
      
      if (downloadToken) {
        // Download via token (for temporary 3D models)
        const blob = await endpoints.download3D(downloadToken);
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `model.${format}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        toast.success('3D model downloaded successfully');
      } else {
        // Direct download from URL
        const response = await fetch(modelUrl);
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `model.${format}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
        toast.success('3D model downloaded successfully');
      }

      if (onDownload) {
        onDownload();
      }
    } catch (error) {
      console.error('Download error:', error);
      toast.error('Failed to download 3D model');
    } finally {
      setIsDownloading(false);
    }
  };

  const handleModelLoad = () => {
    setIsLoading(false);
    setHasError(false);
  };

  const handleModelError = (error: unknown) => {
    console.error('Model loading error:', error);
    setIsLoading(false);
    setHasError(true);
    toast.error('Failed to load 3D model');
  };

  return (
    <div className="w-full h-[600px] bg-gradient-to-br from-gray-50 to-gray-100 rounded-2xl overflow-hidden relative">
      {/* Loading State */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/80 backdrop-blur-sm z-10">
          <div className="flex flex-col items-center gap-3">
            <Loader2 className="h-8 w-8 animate-spin text-gray-700" />
            <p className="text-sm font-medium text-gray-700">Loading 3D model...</p>
          </div>
        </div>
      )}

      {/* Error State */}
      {hasError && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/80 backdrop-blur-sm z-10">
          <div className="flex flex-col items-center gap-3 text-center px-4">
            <AlertCircle className="h-8 w-8 text-red-500" />
            <p className="text-sm font-medium text-gray-900">Failed to load 3D model</p>
            <p className="text-xs text-gray-600">The model file may be corrupted or in an unsupported format</p>
          </div>
        </div>
      )}

      {/* 3D Canvas */}
      <Canvas
        shadows
        camera={{ position: [0, 0, 4], fov: 50 }}
        onCreated={() => handleModelLoad()}
        onError={(error) => handleModelError(error)}
      >
        <SceneContent modelUrl={modelUrl} showGrid={showGrid} />
      </Canvas>

      {/* Control Buttons */}
      <div className="absolute bottom-4 right-4 flex flex-col gap-2">
        <button
          onClick={handleDownload}
          disabled={isDownloading || hasError}
          className="bg-white/90 backdrop-blur-sm p-3 rounded-xl shadow-lg hover:bg-white transition-colors group disabled:opacity-50 disabled:cursor-not-allowed"
          title={isDownloading ? 'Downloading...' : `Download ${format.toUpperCase()}`}
        >
          {isDownloading ? (
            <Loader2 className="h-5 w-5 text-gray-700 animate-spin" />
          ) : (
            <Download className="h-5 w-5 text-gray-700 group-hover:translate-y-0.5 transition-transform" />
          )}
        </button>
        
        <button
          onClick={() => setShowGrid(!showGrid)}
          disabled={hasError}
          className={`backdrop-blur-sm p-3 rounded-xl shadow-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
            showGrid ? 'bg-black text-white' : 'bg-white/90 text-gray-700 hover:bg-white'
          }`}
          title="Toggle Grid"
        >
          <Grid3X3 className="h-5 w-5" />
        </button>
      </div>

      {/* Info Badge */}
      {!hasError && (
        <div className="absolute bottom-4 left-4 bg-white/80 backdrop-blur-sm px-4 py-2 rounded-full text-xs font-medium shadow-lg">
          <span className="text-gray-600">Left-click: Rotate</span>
          <span className="mx-2 text-gray-300">•</span>
          <span className="text-gray-600">Right-click: Pan</span>
          <span className="mx-2 text-gray-300">•</span>
          <span className="text-gray-600">Scroll: Zoom</span>
        </div>
      )}

      {/* Format Badge */}
      <div className="absolute top-4 right-4 bg-black/80 text-white px-3 py-1 rounded-full text-xs font-medium">
        {format.toUpperCase()}
      </div>
    </div>
  );
}

"use client";
import { Canvas } from "@react-three/fiber";
import { useGLTF, Stage, OrbitControls, Grid } from "@react-three/drei";
import { Suspense, useState } from "react";
import { Download, Grid3X3 } from "lucide-react";

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

export default function ModelViewer({ modelUrl }: { modelUrl: string }) {
  const [showGrid, setShowGrid] = useState(false);

  const handleDownloadSTL = () => {
    console.log('STL export - Full implementation requires scene access');
  };

  return (
    <div className="w-full h-[600px] bg-linear-to-br from-gray-50 to-gray-100 rounded-2xl overflow-hidden relative">
      <Canvas
        shadows
        camera={{ position: [0, 0, 4], fov: 50 }}
      >
        <SceneContent modelUrl={modelUrl} showGrid={showGrid} />
      </Canvas>

      {/* Control Buttons */}
      <div className="absolute bottom-4 right-4 flex flex-col gap-2">
        <button
          onClick={handleDownloadSTL}
          className="bg-white/90 backdrop-blur-sm p-3 rounded-xl shadow-lg hover:bg-white transition-colors group"
          title="Download STL"
        >
          <Download className="h-5 w-5 text-gray-700 group-hover:translate-y-0.5 transition-transform" />
        </button>
        
        <button
          onClick={() => setShowGrid(!showGrid)}
          className={`backdrop-blur-sm p-3 rounded-xl shadow-lg transition-colors ${
            showGrid ? 'bg-black text-white' : 'bg-white/90 text-gray-700 hover:bg-white'
          }`}
          title="Toggle Grid"
        >
          <Grid3X3 className="h-5 w-5" />
        </button>
      </div>

      {/* Info Badge */}
      <div className="absolute bottom-4 left-4 bg-white/80 backdrop-blur-sm px-4 py-2 rounded-full text-xs font-medium shadow-lg">
        <span className="text-gray-600">Left-click: Rotate</span>
        <span className="mx-2 text-gray-300">•</span>
        <span className="text-gray-600">Right-click: Pan</span>
        <span className="mx-2 text-gray-300">•</span>
        <span className="text-gray-600">Scroll: Zoom</span>
      </div>
    </div>
  );
}

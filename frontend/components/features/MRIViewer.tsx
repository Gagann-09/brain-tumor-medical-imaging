'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { ZoomIn, ZoomOut, Maximize, RotateCw, Settings2, Eye, EyeOff } from 'lucide-react';
import clsx from 'clsx';
import styles from './MRIViewer.module.css';

interface Layer {
  id: string;
  name: string;
  src?: string; // URL of the image layer (MRI, mask, grad-cam)
  visible: boolean;
  opacity: number;
  type: 'base' | 'mask' | 'heatmap' | 'boundary';
}

interface MRIViewerProps {
  layers: Layer[];
  onLayerChange?: (layers: Layer[]) => void;
  metadata?: Record<string, string>;
}

export const MRIViewer = ({ layers: initialLayers, onLayerChange, metadata }: MRIViewerProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [layers, setLayers] = useState<Layer[]>(initialLayers);
  const [showControls, setShowControls] = useState(false);
  
  // Viewport state
  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Load images
  const [images, setImages] = useState<Record<string, HTMLImageElement>>({});

  useEffect(() => {
    // In a real app, this would load images dynamically.
    // For now, we simulate image loading.
    const loadedImages: Record<string, HTMLImageElement> = {};
    let loadedCount = 0;
    const totalWithSrc = layers.filter(l => l.src).length;

    if (totalWithSrc === 0) return;

    layers.forEach(layer => {
      if (layer.src) {
        const img = new Image();
        img.src = layer.src;
        img.onload = () => {
          loadedImages[layer.id] = img;
          loadedCount++;
          if (loadedCount === totalWithSrc) {
            setImages(loadedImages);
          }
        };
      }
    });
  }, [layers]);

  // Render loop
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas dimensions based on the container (or base image)
    // For simplicity, hardcode a typical size or match container
    const width = containerRef.current?.clientWidth || 800;
    const height = containerRef.current?.clientHeight || 600;
    
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    ctx.clearRect(0, 0, width, height);

    ctx.save();
    // Apply pan and zoom
    ctx.translate(width / 2 + pan.x, height / 2 + pan.y);
    ctx.scale(scale, scale);
    ctx.translate(-width / 2, -height / 2);

    // Draw each visible layer in order
    layers.forEach(layer => {
      if (!layer.visible) return;
      
      const img = images[layer.id];
      if (img) {
        ctx.globalAlpha = layer.opacity;
        
        // Center image
        const x = (width - img.width) / 2;
        const y = (height - img.height) / 2;
        
        ctx.drawImage(img, x, y);
      }
    });

    ctx.restore();
  }, [layers, images, scale, pan]);

  useEffect(() => {
    let animationFrameId: number;
    const render = () => {
      draw();
      animationFrameId = requestAnimationFrame(render);
    };
    render();
    return () => cancelAnimationFrame(animationFrameId);
  }, [draw]);

  // Event Handlers for Pan/Zoom
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const zoomSensitivity = 0.001;
    const delta = -e.deltaY * zoomSensitivity;
    const newScale = Math.min(Math.max(0.1, scale + delta), 10);
    setScale(newScale);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return;
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const resetView = () => {
    setScale(1);
    setPan({ x: 0, y: 0 });
  };

  const toggleLayerVisibility = (id: string) => {
    const newLayers = layers.map(l => l.id === id ? { ...l, visible: !l.visible } : l);
    setLayers(newLayers);
    onLayerChange?.(newLayers);
  };

  const updateLayerOpacity = (id: string, opacity: number) => {
    const newLayers = layers.map(l => l.id === id ? { ...l, opacity } : l);
    setLayers(newLayers);
    onLayerChange?.(newLayers);
  };

  return (
    <div className={styles.container}>
      <div className={styles.toolbar}>
        <div className={styles.toolbarGroup}>
          <button className={styles.iconButton} onClick={() => setScale(s => s * 1.2)} title="Zoom In">
            <ZoomIn size={18} />
          </button>
          <button className={styles.iconButton} onClick={() => setScale(s => s / 1.2)} title="Zoom Out">
            <ZoomOut size={18} />
          </button>
          <button className={styles.iconButton} onClick={resetView} title="Reset View">
            <Maximize size={18} />
          </button>
        </div>
        
        <div className={styles.toolbarGroup}>
          <button 
            className={clsx(styles.iconButton, showControls && styles.active)}
            onClick={() => setShowControls(!showControls)}
            title="Layer Controls"
          >
            <Settings2 size={18} />
          </button>
        </div>
      </div>

      {showControls && (
        <div className={styles.layerControls}>
          <div className={styles.layerControlHeader}>Explainability Layers</div>
          {layers.map(layer => (
            <div key={layer.id} className={styles.layerItem}>
              <button 
                className={styles.iconButton} 
                style={{ width: '24px', height: '24px' }}
                onClick={() => toggleLayerVisibility(layer.id)}
              >
                {layer.visible ? <Eye size={14} /> : <EyeOff size={14} />}
              </button>
              <span style={{ flexGrow: 1 }}>{layer.name}</span>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.01" 
                value={layer.opacity}
                onChange={(e) => updateLayerOpacity(layer.id, parseFloat(e.target.value))}
                className={styles.layerSlider}
                disabled={!layer.visible}
              />
            </div>
          ))}
        </div>
      )}

      <div 
        ref={containerRef}
        className={styles.canvasContainer}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <canvas ref={canvasRef} className={styles.canvas} />
      </div>

      {metadata && (
        <div className={styles.metadata}>
          {Object.entries(metadata).map(([key, value]) => (
            <span key={key}>{key}: {value}</span>
          ))}
          <span>Zoom: {(scale * 100).toFixed(0)}%</span>
        </div>
      )}
    </div>
  );
};

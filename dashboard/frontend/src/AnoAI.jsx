import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

const AnoAI = () => {
  const containerRef = useRef(null);

  useEffect(() => {
    const container = containerRef.current;
    
    // 1. GUARD CLAUSE: If the DOM isn't ready, stop here.
    if (!container) return;

    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    // A full-screen shader has no geometry edges, so antialiasing only costs GPU time.
    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: 'low-power' });

    // The aurora is a soft glow, so rendering it at reduced resolution is visually
    // identical but far cheaper (the shader runs a 35-step loop for every pixel).
    const RENDER_SCALE = 0.5;
    const resolution = new THREE.Vector2();
    const applySize = () => {
      renderer.setPixelRatio(RENDER_SCALE);
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.getDrawingBufferSize(resolution);
    };
    applySize();

    container.appendChild(renderer.domElement);

    const material = new THREE.ShaderMaterial({
      uniforms: {
        iTime: { value: 0 },
        iResolution: { value: resolution }
      },
      vertexShader: `
        void main() {
          gl_Position = vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float iTime;
        uniform vec2 iResolution;

        #define NUM_OCTAVES 3

        float rand(vec2 n) {
          return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
        }

        float noise(vec2 p) {
          vec2 ip = floor(p);
          vec2 u = fract(p);
          u = u*u*(3.0-2.0*u);

          float res = mix(
            mix(rand(ip), rand(ip + vec2(1.0, 0.0)), u.x),
            mix(rand(ip + vec2(0.0, 1.0)), rand(ip + vec2(1.0, 1.0)), u.x), u.y);
          return res * res;
        }

        float fbm(vec2 x) {
          float v = 0.0;
          float a = 0.3;
          vec2 shift = vec2(100);
          mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
          for (int i = 0; i < NUM_OCTAVES; ++i) {
            v += a * noise(x);
            x = rot * x * 2.0 + shift;
            a *= 0.4;
          }
          return v;
        }

        void main() {
          vec2 shake = vec2(sin(iTime * 1.2) * 0.005, cos(iTime * 2.1) * 0.005);
          vec2 p = ((gl_FragCoord.xy + shake * iResolution.xy) - iResolution.xy * 0.5) / iResolution.y * mat2(6.0, -4.0, 4.0, 6.0);
          vec2 v;
          vec4 o = vec4(0.0);

          float f = 2.0 + fbm(p + vec2(iTime * 5.0, 0.0)) * 0.5;

          for (float i = 0.0; i < 35.0; i++) {
            v = p + cos(i * i + (iTime + p.x * 0.08) * 0.025 + i * vec2(13.0, 11.0)) * 3.5 + vec2(sin(iTime * 3.0 + i) * 0.003, cos(iTime * 3.5 - i) * 0.003);
            float tailNoise = fbm(v + vec2(iTime * 0.5, i)) * 0.3 * (1.0 - (i / 35.0));
            vec4 auroraColors = vec4(
              0.1 + 0.3 * sin(i * 0.2 + iTime * 0.4),
              0.3 + 0.5 * cos(i * 0.3 + iTime * 0.5),
              0.7 + 0.3 * sin(i * 0.4 + iTime * 0.3),
              1.0
            );
            vec4 currentContribution = auroraColors * exp(sin(i * i + iTime * 0.8)) / length(max(v, vec2(v.x * f * 0.015, v.y * 1.5)));
            float thinnessFactor = smoothstep(0.0, 1.0, i / 35.0) * 0.6;
            o += currentContribution * (1.0 + tailNoise * 0.8) * thinnessFactor;
          }

          // Added a slight dark tint so the Glassmorphism UI pops out better
          o = tanh(pow(o / 100.0, vec4(1.6))) * 0.8;
          gl_FragColor = o * 1.5;
        }
      `
    });

    const geometry = new THREE.PlaneGeometry(2, 2);
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    let frameId = null;
    let last = performance.now();
    const animate = (now = performance.now()) => {
      // Advance by real elapsed time (same speed as before at 60 fps, but not
      // faster on 120/144 Hz screens); clamp so a paused tab doesn't jump ahead.
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      material.uniforms.iTime.value += dt * 0.96;
      renderer.render(scene, camera);
      frameId = reducedMotion ? null : requestAnimationFrame(animate);
    };
    animate();

    // Stop rendering entirely while the tab is hidden
    const handleVisibility = () => {
      if (document.hidden) {
        if (frameId !== null) cancelAnimationFrame(frameId);
        frameId = null;
      } else if (frameId === null && !reducedMotion) {
        last = performance.now();
        frameId = requestAnimationFrame(animate);
      }
    };
    document.addEventListener('visibilitychange', handleVisibility);

    const handleResize = () => {
      applySize();
      if (reducedMotion) renderer.render(scene, camera);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup function
    return () => {
      if (frameId !== null) cancelAnimationFrame(frameId);
      document.removeEventListener('visibilitychange', handleVisibility);
      window.removeEventListener('resize', handleResize);
      
      // 2. SAFE REMOVAL: Only remove if the container and canvas both still exist
      if (container && renderer.domElement && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      
      geometry.dispose();
      material.dispose();
      renderer.dispose();
    };
  }, []);

  return (
    // Styling this to take up the full width/height of the background layer
    <div ref={containerRef} className="w-full h-full overflow-hidden bg-black" />
  );
};

export default AnoAI;
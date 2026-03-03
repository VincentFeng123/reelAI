"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";

const VERTEX_SHADER = `
void main() {
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const FRAGMENT_SHADER = `
precision highp float;
uniform float u_time;
uniform vec2 u_res;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  mat2 rot = mat2(0.866, 0.5, -0.5, 0.866);
  for (int i = 0; i < 5; i++) {
    v += a * noise(p);
    p = rot * p * 2.0;
    a *= 0.5;
  }
  return v;
}

vec3 ramp(float t) {
  vec3 c1 = vec3(0.005, 0.008, 0.035);
  vec3 c2 = vec3(0.06, 0.10, 0.26);
  vec3 c3 = vec3(0.28, 0.42, 0.72);
  vec3 c4 = vec3(0.72, 0.83, 0.98);
  vec3 c5 = vec3(0.97, 0.98, 1.00);
  t = clamp(t, 0.0, 2.0);
  vec3 a = mix(c1, c2, smoothstep(0.0, 0.32, t));
  vec3 b = mix(a, c3, smoothstep(0.22, 0.68, t));
  vec3 c = mix(b, c4, smoothstep(0.55, 1.15, t));
  return mix(c, c5, smoothstep(0.95, 1.55, t));
}

void main() {
  vec2 uv = gl_FragCoord.xy / u_res.xy;
  float aspect = u_res.x / max(u_res.y, 1.0);
  vec2 p = vec2(uv.x * aspect, uv.y);

  float t = u_time * 0.18;

  vec2 source = vec2(aspect * 1.08, 0.50);
  vec2 vortexA = vec2(aspect * 0.52, 0.32);
  vec2 vortexB = vec2(aspect * 0.40, 0.80);

  vec2 q = p;

  float w1 = fbm(q * 1.6 + vec2(-t * 0.55, t * 0.08));
  float w2 = fbm(q * 3.0 + vec2(-t * 0.9, t * 0.30));

  vec2 dA = q - vortexA;
  float rA = length(dA) + 0.001;
  vec2 tangA = vec2(-dA.y, dA.x) / rA;
  float spinA = exp(-rA * 2.8);

  vec2 dB = q - vortexB;
  float rB = length(dB) + 0.001;
  vec2 tangB = vec2(-dB.y, dB.x) / rB;
  float spinB = exp(-rB * 2.4);

  q += tangA * (0.10 + 0.07 * w1) * spinA;
  q -= tangB * (0.05 + 0.03 * w2) * spinB;

  float dist = distance(q, source);
  float sourceInfluence = smoothstep(1.3, 0.04, dist);
  q += (vec2(w1, w2) - 0.5) * (0.09 * sourceInfluence);

  float fromRight = max(0.0, source.x - q.x);
  float along = clamp(fromRight / (aspect * 1.08), 0.0, 1.0);

  float centerline = source.y
    - 0.06 * along
    + 0.035 * sin(along * 5.0 - t * 1.2)
    + (w1 - 0.5) * 0.05 * along;

  float width = mix(0.58, 0.12, pow(along, 0.65));
  float axis = abs(q.y - centerline);

  float radial = 1.0 / (dist * 1.1 + 0.035);
  radial = pow(radial, 1.55);

  float haze = exp(-axis / (width * 3.0 + 0.02)) * (0.42 + 0.58 * (1.0 - along));

  float streakField = fbm(vec2(fromRight * 3.5 - t * 1.5, (q.y - centerline) * 6.0 + t * 0.8));
  float streaks = smoothstep(0.35, 0.85, streakField);
  streaks *= exp(-axis / (width * 1.6 + 0.01));

  float arc = exp(-pow(length((q - vortexA) * vec2(1.0, 1.4)) - 0.20, 2.0) * 100.0);
  float hook = exp(-length((q - vortexA) * vec2(0.8, 1.3)) * 6.0);

  float plumeMask = smoothstep(0.0, 0.05, fromRight);

  float field = radial * (0.65 + 0.45 * fbm(q * 2.0 + vec2(-t * 0.25, t * 0.12)));
  field += haze * 1.15 * plumeMask;
  field += streaks * (0.75 - 0.30 * along) * plumeMask;
  field += arc * 0.30 + hook * 0.20;

  float whiteMass = exp(-distance(q, vec2(aspect * 1.05, 0.50)) * 1.4);
  field += whiteMass * 2.4;

  float glow2 = exp(-distance(q, vec2(aspect * 0.90, 0.46)) * 2.2);
  field += glow2 * 0.55;

  float glow3 = exp(-distance(q, vec2(aspect * 0.85, 0.58)) * 2.5);
  field += glow3 * 0.35;

  vec3 col = ramp(field);
  col += vec3(1.0) * pow(max(field - 0.65, 0.0), 1.25) * 0.85;

  float leftDark = smoothstep(0.0, 0.50, uv.x);
  col *= mix(0.28, 1.0, leftDark);

  float topDark = smoothstep(0.85, 0.03, uv.y) * smoothstep(0.0, 0.50, uv.x);
  col *= 1.0 - topDark * 0.30;

  float bottomPocket = exp(-distance(uv, vec2(0.18, 0.98)) * 3.2);
  col *= 1.0 - bottomPocket * 0.35;

  vec2 v = uv * 2.0 - 1.0;
  float vignette = 1.0 - 0.45 * dot(v * vec2(0.75, 0.95), v);
  vignette = clamp(vignette, 0.0, 1.0);
  col *= vignette;

  col += vec3(0.04, 0.06, 0.14) * smoothstep(0.08, 0.65, 1.0 - uv.x) * 0.15;

  float grain = (hash(gl_FragCoord.xy + u_time * 48.0) - 0.5) * 0.005;
  col += grain;

  col = clamp(col, 0.0, 1.0);
  gl_FragColor = vec4(col, 1.0);
}
`;

type VolumetricUniforms = {
  u_time: { value: number };
  u_res: { value: THREE.Vector2 };
};

export function VolumetricGlowBackground() {
  const mountRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount || typeof window === "undefined") {
      return;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });

    renderer.setClearColor(0x000000, 1);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    const canvas = renderer.domElement;
    mount.appendChild(canvas);

    const uniforms: VolumetricUniforms = {
      u_time: { value: 0 },
      u_res: { value: new THREE.Vector2(1, 1) },
    };

    const material = new THREE.ShaderMaterial({
      uniforms,
      vertexShader: VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      depthWrite: false,
      depthTest: false,
    });

    const geometry = new THREE.PlaneGeometry(2, 2);
    const quad = new THREE.Mesh(geometry, material);
    scene.add(quad);

    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));

    const bloomPass = new UnrealBloomPass(new THREE.Vector2(1, 1), 1.6, 0.92, 0.06);
    composer.addPass(bloomPass);

    const resize = () => {
      const width = Math.max(1, mount.clientWidth || window.innerWidth);
      const height = Math.max(1, mount.clientHeight || window.innerHeight);
      renderer.setSize(width, height, false);
      composer.setSize(width, height);
      bloomPass.setSize(width, height);
      uniforms.u_res.value.set(width, height);
    };

    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const motionScale = prefersReducedMotion ? 0.45 : 1;
    const startTime = performance.now();
    let rafId = 0;

    const tick = (now: number) => {
      uniforms.u_time.value = ((now - startTime) / 1000) * motionScale;
      composer.render();
      rafId = window.requestAnimationFrame(tick);
    };

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(mount);
    window.addEventListener("resize", resize);
    resize();
    rafId = window.requestAnimationFrame(tick);

    return () => {
      window.cancelAnimationFrame(rafId);
      resizeObserver.disconnect();
      window.removeEventListener("resize", resize);

      geometry.dispose();
      material.dispose();
      composer.dispose();
      renderer.dispose();
      renderer.forceContextLoss();

      if (canvas.parentNode) {
        canvas.parentNode.removeChild(canvas);
      }
    };
  }, []);

  return <div ref={mountRef} className="volumetric-glow-canvas" aria-hidden="true" />;
}

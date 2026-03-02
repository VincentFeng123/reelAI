"use client";

import { useEffect, useRef } from "react";

const VERTEX_SHADER = `#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

attribute vec2 aPosition;
varying vec2 vUv;

void main() {
    vUv = 0.5 * aPosition + 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}
`;

const RENDER_FRAGMENT_SHADER = `#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

varying vec2 vUv;
uniform float uTime;
uniform vec2 uResolution;
uniform float uLogoScale;
uniform vec2 uOffset;
uniform sampler2D uNoiseTexture;
uniform sampler2D uLogoTexture;
uniform sampler2D uTrailTexture;

//Star brightness
#define STAR 5.0
//Flare brightness
#define FLARE 4.0
// Star color
#define COLOR vec3(0.2, 0.3, 0.8)

// Star turbulence parameters
#define STAR_NUM 12.0
#define STAR_AMP 0.5
#define STAR_SPEED 0.01
#define STAR_VEL vec2(1.0, 0.0)
#define STAR_FREQ 8.0
#define STAR_EXP 1.5

// Logo size (relative to screen-y)
// Set in both shaders!
#define LOGO_SCALE 0.5
// Aspect ratio (w / h)
#define LOGO_RATIO 2.08

// Glow strength
#define GLOW_STRENGTH 12.0
// Glow colors
#define GLOW_RED vec3(0.5, 0.2, 0.2)
#define GLOW_BLUE vec3(0.3, 0.3, 0.6)
// Glow turbulence strength
#define GLOW_TURBULENCE 0.4
// Glow chromatic aberration
#define GLOW_TINT 3.0

// Light vertical falloff exponent (higher = narrower)
#define LIGHT_EXP 30.0

// Trail RGB falloff exponents (higher = darker)
#define TRAIL_EXP vec3(1.4, 1.2, 1.0)
// Trail strength (0.0 = no trail, 1.0 = full strength)
#define TRAIL_STRENGTH 0.4

// Dither intensity
#define DITHER 0.01
// Dither texture resolution
#define DITHER_RES 64.0

// Gamma encoding (with Gamma = 2.0)
vec3 gamma_encode(vec3 lrgb) {
    return sqrt(lrgb);
}

// Turbulence waves
vec2 turbulence(vec2 p, float freq, float num) {
    mat2 rot = mat2(0.6, -0.8, 0.8, 0.6);
    vec2 turb = vec2(0.0);
    for (float i = 0.0; i < STAR_NUM; i++) {
        if (i >= num) break;

        vec2 pos = p + turb + STAR_SPEED * i * uTime * STAR_VEL;
        float phase = freq * (pos * rot).y + STAR_SPEED * uTime * freq;
        turb += rot[0] * sin(phase) / freq;
        rot *= mat2(0.6, -0.8, 0.8, 0.6);
        freq *= STAR_EXP;
    }
    return turb;
}

// Star background
vec3 star(inout vec2 p) {

    //Horizontal stretching factor (1 / scale)
    #define STAR_STRETCH 0.7
    #define STAR_CURVE 0.5

    // Signed range [-1, 1]
    vec2 suv = p * 2.0 - 1.0;
    // Coordinates relative to right side
    vec2 right = suv - vec2(1.0, 0.0);

    // Aspect corrected
    right.x *= STAR_STRETCH * uResolution.x / uResolution.y;
    // Apply turbulence
    // Variable turbulence intensity
    float factor = 1.0 + 0.4 * sin(9.0 * suv.y) * sin(5.0 * (suv.x + 5.0 * uTime * STAR_SPEED));
    vec2 turb = right + factor * STAR_AMP * turbulence(right, STAR_FREQ, STAR_NUM);
    // Shift top and bottom edges
    turb.x -= STAR_CURVE * suv.y * suv.y;

    // Attenuate slower inside
    float fade = max(4.0 * suv.y * suv.y - suv.x + 1.2, 0.001);
    float atten = fade * max(0.5 * turb.x, -turb.x);

    // Flare time
    float ft = 0.4 * uTime;
    // Flare position
    vec2 fp = 8.0 * (turb + 0.5 * STAR_VEL * ft);
    fp *= mat2(0.4, -0.3, 0.3, 0.4);
    // Flare
    float f = cos(fp.x) * sin(fp.y) - 0.5;
    // Flare brightness
    float flare = f * f + 0.5 * suv.y * suv.y - 1.5 * turb.x + 0.6 * cos(0.42 * ft + 1.6 * turb.y) * cos(0.31 * ft - turb.y);

    // Star brightness
    vec3 col = 0.1 * COLOR * (STAR / (atten * atten) + FLARE / (flare * flare));

    // Chroma phase shift
    const vec3 chrom = vec3(0.0, 0.1, 0.2);
    // Color rays
    col *= exp(p.x *
                cos(turb.y * 5.0 + 0.4 * (uTime + turb.x * 1.0) + chrom) *
                cos(turb.y * 7.0 - 0.5 * (uTime - turb.x * 1.5) + chrom) *
                cos(turb.y * 9.0 + 0.6 * (uTime + turb.x * 2.0) + chrom)
        );

    return col;
}

void main() {
    // Dither uv
    vec2 duv = 0.9 * gl_FragCoord.xy / DITHER_RES * mat2(0.8, -0.6, 0.6, 0.8);
    // Sample signed dithering [-0.5, +0.5]
    float dither = texture2D(uNoiseTexture, duv).r - 0.5;

    // Capped aspect ratio
    vec2 ratio = min(uResolution.yx / uResolution.xy, 1.0);
    // Sample trail texture
    vec4 trailTex = texture2D(uTrailTexture, vUv);

    // Signed screen uvs [-1, +1]
    vec2 suv = vUv * 2.0 - 1.0;

    // Compute logo scale (aspect ratio corrected)
    vec2 scale = max(uLogoScale, 1.0 - (LOGO_RATIO / 4.0)) * ratio * vec2(LOGO_RATIO, -1.0);
    // Normalized logo uvs
    vec2 logoUv = 0.5 + (vUv - 0.5) / scale;

    // Logo texture + turbulent samples
    vec4 logo = vec4(0);
    vec4 logoTurb = vec4(0);
    // Signed direction vector from logo
    vec2 dir = vec2(0);
    // Glow intensity
    float glow = 0.0;
    // UV distortions
    vec2 distort = uOffset;

    // Bounding box check
    if (logoUv.x >= 0.0 && logoUv.x <= 1.0 && logoUv.y >= 0.0 && logoUv.y <= 1.0) {
        // Sample logo
        logo = texture2D(uLogoTexture, logoUv);

        // Direction vector (flipped x)
        dir = (logo.rg - 0.6);
        dir.x = -dir.x;

        // Twist around logo
        vec2 shift = -2.0 * vec2(dir.y, -dir.x) * dir.y * logo.b;
        // Trail distortion (0.1 seems reasonable)
        shift += 0.1 * (1.0 - logo.b) * (trailTex.rg - 0.5) * trailTex.b * ratio;
        // Correct for ratio and shift
        vec2 logoT = (logoUv) * vec2(LOGO_RATIO, 1.0) + shift;
        // Add turbulence
        logoT += (1.0 - logo.b) * turbulence(logoT, 40.0, 6.0);
        // Convert back to normalized uvs
        logoUv = (logoT - shift) / vec2(LOGO_RATIO, 1.0);

        // Sample logo turbulence
        logoTurb = texture2D(uLogoTexture, logoUv);
        // Mix with logo alpha
        logoTurb.b = mix(logo.b, logoTurb.b, GLOW_TURBULENCE);

        // Glow
        // Horizontal fade
        float xx = logoUv.x;
        // Vertical fade
        float yy = (logoUv.y - 0.5);
        // Glow intensity
        glow = max(logoTurb.b - (xx * xx + 8.0 * yy * yy) * logoTurb.b, 0.0);

        // Distort round logo
        distort += dir * logo.b * (1.0 - logo.b);
    }

    // Star
    vec2 starUv = vUv + distort;
    // Add trail distortion
    starUv += 0.3 * (trailTex.rg - 0.5) * trailTex.b * ratio;
    // Get star color
    vec3 col = star(starUv);

    // Vertical vignette
    float vig = 1.0 - abs(suv.y);
    // Horizontal fade
    vig *= 0.5 + 0.5 * suv.x;
    // Apply vignette
    col *= vig * vig;

    // Tonemap and gamma encode
    col /= 1.0 + col;
    col = clamp(col, 0.0, 1.0);
    col = gamma_encode(col);

    // Light gradient
    float yy = suv.y + 0.03;
    yy = max(1.0 - 1e1 * yy * yy / max(0.5 + 1.5 * starUv.x, 0.1), 0.0);
    float light = max(0.5 + 0.5 * starUv.x, 0.0) * yy;
    light += 2.0 * (1.0 - light) * glow;

    // Rim
    float tint = GLOW_TINT * dir.x * glow;
    vec3 hue = mix(GLOW_RED, GLOW_BLUE, 1.0 + suv.x + tint);
    float alpha = 1.0 - (1.0 - pow(yy, LIGHT_EXP)) * glow;
    vec3 rim = GLOW_STRENGTH * light * light * light * light * alpha * (0.5 + 0.5 * suv.x) * hue;

    // Rim tone mapping
    rim /= (1.0 + rim);
    // Add rim glow
    col += (1.0 - col) * rim * rim;
    // Add trail
    col += TRAIL_STRENGTH * hue * pow(trailTex.aaa, TRAIL_EXP);
    // Logo mask
    float a = smoothstep(1.0, 0.2, logo.a);
    col.rgb = a * col.rgb + (1.0 - a);

    // Apply dithering
    col += DITHER * dither;

    gl_FragColor = vec4(col, 1.0);
}
`;

const TRAIL_FRAGMENT_SHADER = `#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

varying vec2 vUv;
uniform float uTime;
uniform float uDeltaTime;
uniform float uLogoScale;
uniform vec2 uMouse;
uniform vec2 uMouseVelocity;
uniform vec2 uResolution;
uniform sampler2D uNoiseTexture;
uniform sampler2D uPreviousFrame;
uniform sampler2D uLogoTexture;

// Trail falloff (higher = narrower)
#define TRAIL_FALLOFF 9000.0
// Fade exponents (velocity x, velocity y, motion, alpha)
#define FADE_EXP vec4(0.02, 0.02, 0.1, 0.1)

// Base scrolling speed
#define SCROLL_SPEED 0.0005
// Turbulent distortion speed
#define DISTORT_SPEED 0.02

// Logo twirl strength
#define LOGO_TWIRL 0.4
// Logo pull strength
#define LOGO_PULL 0.1

// Logo size (relative to screen-y)
// Set in both shaders!
#define LOGO_SCALE 0.5
// Aspect ratio (w / h)
#define LOGO_RATIO 2.08

// Turbulence parameters
#define TURB_NUM 8.0
#define TURB_AMP 0.6
#define TURB_SPEED 0.5
#define TURB_VEL vec2(0.1, 0.0)
#define TURB_FREQ 50.0
#define TURB_EXP 1.3

vec2 turbulence(vec2 p) {
    mat2 rot = mat2(0.6, -0.8, 0.8, 0.6);
    vec2 turb = vec2(0.0);
    float freq = TURB_FREQ;
    for (float i = 0.0; i < TURB_NUM; i++) {
        vec2 pos = p + TURB_SPEED * i * uTime * TURB_VEL;
        float phase = freq * (pos * rot).y + TURB_SPEED * uTime * freq * 0.1;
        turb += rot[0] * sin(phase) / freq;
        rot *= mat2(0.6, -0.8, 0.8, 0.6);
        freq *= TURB_EXP;
    }
    return turb;
}

void main() {
    // Capped aspect ratio
    vec2 ratio = min(uResolution.yx / uResolution.xy, 1.0);

    // Compute logo scale (aspect ratio corrected)
    vec2 scale = max(uLogoScale, 1.0 - (LOGO_RATIO / 4.0)) * ratio * vec2(LOGO_RATIO, -1.0);
    // Normalized logo uvs
    vec2 logoUV = 0.5 + (vUv - 0.5) / scale;
    // Sample logo
    vec4 logo = vec4(0);
    if (logoUV.x >= 0.0 && logoUV.x <= 1.0 && logoUV.y >= 0.0 && logoUV.y <= 1.0) {
        logo = texture2D(uLogoTexture, logoUV);
    }

    // Delta rate
    float delta = 144.0 * uDeltaTime;
    // Scroll velocity
    vec2 scroll = SCROLL_SPEED * vec2(1.0, vUv.y - 0.5) * ratio;
    // Turbulent distortion vector
    vec2 turb = turbulence((vUv + scroll) / ratio);
    // Distortion velocity
    vec2 distort = DISTORT_SPEED * turb;
    // Add logo twirl and pull
    distort -= LOGO_TWIRL * (logo.rg - 0.6) * mat2(0, -1, 1, 0) * (logo.g - 0.5) * logo.b;
    distort -= LOGO_PULL * (logo.rg - 0.6) * logo.b * logo.b;
    // Distorted UVs
    vec2 distortedUv = vUv + delta * scroll + delta * distort * ratio;

    // Sample previous frame with distortion
    vec4 prev = texture2D(uPreviousFrame, distortedUv);

    // Create trail effect based on mouse velocity and position
    // Mouse trail start and end points
    vec2 trailA = vUv + 0.01 * delta * turb * ratio - uMouse;
    vec2 trailB = -uMouseVelocity;
    // Trail distance squared
    float trailD = dot(trailB, trailB);
    // Vector to nearest trail point
    vec2 trailDif = trailA / ratio;
    // Falloff
    float falloff = 0.0;
    if (trailD > 0.0) {
        // Normalized segment factor
        float f = clamp(dot(trailA, trailB) / trailD, 0.0, 1.0);
        // Trail difference to uvs
        trailDif -= f * trailB / ratio;
        // Falloff
        falloff = (1.0 - logo.b) / (1.0 + TRAIL_FALLOFF * dot(trailDif, trailDif));
        // Normalize falloff
        falloff *= min(trailD / (0.001 + trailD), 1.0);
    }

    // Compute brightness value
    vec2 suv = (uMouse - uMouseVelocity) * 2.0 - 1.0;
    // Vignette
    float vig = 1.0 - abs(suv.y);
    // Horizontal fade
    vig *= 0.5 + 0.5 * suv.x;

    // Sample noise for dithered falloff
    vec2 nuv = gl_FragCoord.xy / 64.0 + uTime * vec2(7.1, 9.1);
    float noise = texture2D(uNoiseTexture, nuv).r;

    // Falloff exponents
    vec4 fade = pow(vec4(noise), FADE_EXP);
    // Delta timed decay
    fade = exp(-2.0 * fade * uDeltaTime);
    // Mix previous frame with current trail
    vec4 decay = mix(vec4(0.5, 0.5, 0.0, 0.0), prev, fade);

    //Set output color
    vec4 col = decay;

    //Trail velocity
    vec2 vel = (-trailB) / (0.01 + length(trailB));
    //Add trail velocity (smooth blended)
    col.rg -= (0.5 - abs(decay.rg - 0.5)) * (falloff * vel);
    //Add trail falloff (smooth blended)
    col.ba += falloff * (1.0 - decay.ba) * vec2(1.0, vig * vig);

    //Stochastic dithering
    col += (noise - 0.5) / 255.0;
    gl_FragColor = col;
}
`;

type GL = WebGLRenderingContext | WebGL2RenderingContext;

type RenderUniforms = {
  time: WebGLUniformLocation | null;
  resolution: WebGLUniformLocation | null;
  logoScale: WebGLUniformLocation | null;
  offset: WebGLUniformLocation | null;
  noiseTexture: WebGLUniformLocation | null;
  logoTexture: WebGLUniformLocation | null;
  trailTexture: WebGLUniformLocation | null;
};

type TrailUniforms = {
  time: WebGLUniformLocation | null;
  deltaTime: WebGLUniformLocation | null;
  logoScale: WebGLUniformLocation | null;
  mouse: WebGLUniformLocation | null;
  mouseVelocity: WebGLUniformLocation | null;
  resolution: WebGLUniformLocation | null;
  noiseTexture: WebGLUniformLocation | null;
  previousFrame: WebGLUniformLocation | null;
  logoTexture: WebGLUniformLocation | null;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function createShader(gl: GL, type: number, source: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) {
    return null;
  }
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    return shader;
  }
  console.error("Shader compile error:", gl.getShaderInfoLog(shader));
  gl.deleteShader(shader);
  return null;
}

function createProgram(gl: GL, vertexSource: string, fragmentSource: string): WebGLProgram | null {
  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  if (!vertexShader || !fragmentShader) {
    if (vertexShader) {
      gl.deleteShader(vertexShader);
    }
    if (fragmentShader) {
      gl.deleteShader(fragmentShader);
    }
    return null;
  }

  const program = gl.createProgram();
  if (!program) {
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
    return null;
  }

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);

  if (gl.getProgramParameter(program, gl.LINK_STATUS)) {
    return program;
  }

  console.error("Program link error:", gl.getProgramInfoLog(program));
  gl.deleteProgram(program);
  return null;
}

function createRenderTexture(gl: GL, width: number, height: number): WebGLTexture | null {
  const texture = gl.createTexture();
  if (!texture) {
    return null;
  }
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, Math.max(1, width), Math.max(1, height), 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return texture;
}

function createFramebuffer(gl: GL, texture: WebGLTexture): WebGLFramebuffer | null {
  const framebuffer = gl.createFramebuffer();
  if (!framebuffer) {
    return null;
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    console.error("Framebuffer not complete. Status:", status);
    gl.deleteFramebuffer(framebuffer);
    return null;
  }
  return framebuffer;
}

function loadTexture(gl: GL, path: string, repeat: boolean, maxTextureSize: number): Promise<WebGLTexture> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const texture = gl.createTexture();

    if (!texture) {
      reject(new Error("Failed to create texture"));
      return;
    }

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([128, 128, 128, 255]));

    image.onload = () => {
      let sourceWidth = image.width;
      let sourceHeight = image.height;
      gl.bindTexture(gl.TEXTURE_2D, texture);

      if (sourceWidth > maxTextureSize || sourceHeight > maxTextureSize) {
        const scale = Math.min(maxTextureSize / sourceWidth, maxTextureSize / sourceHeight);
        sourceWidth = Math.max(1, Math.floor(sourceWidth * scale));
        sourceHeight = Math.max(1, Math.floor(sourceHeight * scale));
        const offscreen = document.createElement("canvas");
        offscreen.width = sourceWidth;
        offscreen.height = sourceHeight;
        const context = offscreen.getContext("2d");
        if (!context) {
          gl.deleteTexture(texture);
          reject(new Error(`Failed to resize texture ${path}`));
          return;
        }
        context.drawImage(image, 0, 0, sourceWidth, sourceHeight);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, offscreen);
      } else {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
      }

      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, repeat ? gl.REPEAT : gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, repeat ? gl.REPEAT : gl.CLAMP_TO_EDGE);
      resolve(texture);
    };

    image.onerror = () => {
      gl.deleteTexture(texture);
      reject(new Error(`Failed to load texture ${path}`));
    };

    image.src = path;
  });
}

export function VolumetricLightBackground() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) {
      return;
    }

    const gl =
      ((canvas.getContext("webgl2", {
        alpha: true,
        antialias: false,
        powerPreference: "high-performance",
        premultipliedAlpha: false,
      }) as GL | null) ??
        (canvas.getContext("webgl", {
          alpha: true,
          antialias: false,
          powerPreference: "high-performance",
          premultipliedAlpha: false,
        }) as GL | null));

    if (!gl) {
      console.error("WebGL is not supported by this browser.");
      return;
    }

    const ownedBuffers: WebGLBuffer[] = [];
    const ownedTextures: WebGLTexture[] = [];
    const ownedFramebuffers: WebGLFramebuffer[] = [];
    const ownedPrograms: WebGLProgram[] = [];

    const renderProgram = createProgram(gl, VERTEX_SHADER, RENDER_FRAGMENT_SHADER);
    const trailProgram = createProgram(gl, VERTEX_SHADER, TRAIL_FRAGMENT_SHADER);
    if (!renderProgram || !trailProgram) {
      return;
    }
    ownedPrograms.push(renderProgram, trailProgram);

    const quadBuffer = gl.createBuffer();
    if (!quadBuffer) {
      return;
    }
    ownedBuffers.push(quadBuffer);

    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    const bindPositionAttribute = (program: WebGLProgram) => {
      gl.useProgram(program);
      const location = gl.getAttribLocation(program, "aPosition");
      if (location < 0) {
        return;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);
    };
    bindPositionAttribute(renderProgram);
    bindPositionAttribute(trailProgram);

    const renderUniforms: RenderUniforms = {
      time: gl.getUniformLocation(renderProgram, "uTime"),
      resolution: gl.getUniformLocation(renderProgram, "uResolution"),
      logoScale: gl.getUniformLocation(renderProgram, "uLogoScale"),
      offset: gl.getUniformLocation(renderProgram, "uOffset"),
      noiseTexture: gl.getUniformLocation(renderProgram, "uNoiseTexture"),
      logoTexture: gl.getUniformLocation(renderProgram, "uLogoTexture"),
      trailTexture: gl.getUniformLocation(renderProgram, "uTrailTexture"),
    };

    const trailUniforms: TrailUniforms = {
      time: gl.getUniformLocation(trailProgram, "uTime"),
      deltaTime: gl.getUniformLocation(trailProgram, "uDeltaTime"),
      resolution: gl.getUniformLocation(trailProgram, "uResolution"),
      logoScale: gl.getUniformLocation(trailProgram, "uLogoScale"),
      mouse: gl.getUniformLocation(trailProgram, "uMouse"),
      mouseVelocity: gl.getUniformLocation(trailProgram, "uMouseVelocity"),
      noiseTexture: gl.getUniformLocation(trailProgram, "uNoiseTexture"),
      logoTexture: gl.getUniformLocation(trailProgram, "uLogoTexture"),
      previousFrame: gl.getUniformLocation(trailProgram, "uPreviousFrame"),
    };

    const setUniform1f = (location: WebGLUniformLocation | null, value: number) => {
      if (location !== null) {
        gl.uniform1f(location, value);
      }
    };

    const setUniform1i = (location: WebGLUniformLocation | null, value: number) => {
      if (location !== null) {
        gl.uniform1i(location, value);
      }
    };

    const setUniform2f = (location: WebGLUniformLocation | null, x: number, y: number) => {
      if (location !== null) {
        gl.uniform2f(location, x, y);
      }
    };

    let trailTextures: [WebGLTexture, WebGLTexture] | null = null;
    let trailFramebuffers: [WebGLFramebuffer, WebGLFramebuffer] | null = null;
    let pingPongIndex = 0;
    let qualityScale = 1.0;
    const minQualityScale = 0.5;
    const maxQualityScale = 1.0;

    let noiseTexture: WebGLTexture | null = null;
    let logoTexture: WebGLTexture | null = null;
    let currentLogoPath = "";

    let disposed = false;
    let rafId = 0;
    let lastTime = 0;
    let frameCount = 0;
    let fpsWindowStart = performance.now();
    let qualityCheckStart = performance.now();
    let renderWidth = 1;
    let renderHeight = 1;

    const mouse = [0.5, 0.5];
    const mouseSmoothed = [0.5, 0.5];

    const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;

    const selectLogoPath = () => {
      if (canvas.width >= 900) {
        return "/images/logo.png";
      }
      if (canvas.width >= 450) {
        return "/images/logoHalf.png";
      }
      return "/images/logoQuat.png";
    };

    const applyStaticUniforms = () => {
      const offsetX = -0.3 * (1 - Math.min(canvas.width / 1200, 1));
      gl.useProgram(renderProgram);
      setUniform2f(renderUniforms.resolution, canvas.width, canvas.height);
      setUniform1f(renderUniforms.logoScale, 0.5);
      setUniform2f(renderUniforms.offset, offsetX, 0);

      gl.useProgram(trailProgram);
      setUniform2f(trailUniforms.resolution, canvas.width, canvas.height);
      setUniform1f(trailUniforms.logoScale, 0.5);
    };

    const ensureTrailTargets = () => {
      const targetWidth = Math.max(1, Math.floor(canvas.width * qualityScale));
      const targetHeight = Math.max(1, Math.floor(canvas.height * qualityScale));
      renderWidth = targetWidth;
      renderHeight = targetHeight;

      if (!trailTextures || !trailFramebuffers) {
        const textureA = createRenderTexture(gl, targetWidth, targetHeight);
        const textureB = createRenderTexture(gl, targetWidth, targetHeight);
        if (!textureA || !textureB) {
          return;
        }
        ownedTextures.push(textureA, textureB);

        const framebufferA = createFramebuffer(gl, textureA);
        const framebufferB = createFramebuffer(gl, textureB);
        if (!framebufferA || !framebufferB) {
          return;
        }
        ownedFramebuffers.push(framebufferA, framebufferB);
        trailTextures = [textureA, textureB];
        trailFramebuffers = [framebufferA, framebufferB];
      } else {
        for (const texture of trailTextures) {
          gl.bindTexture(gl.TEXTURE_2D, texture);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, targetWidth, targetHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        }
      }
    };

    const resizeCanvas = () => {
      const width = Math.max(1, Math.floor(container.clientWidth));
      const height = Math.max(1, Math.floor(container.clientHeight));
      canvas.width = width;
      canvas.height = height;
      ensureTrailTargets();
      applyStaticUniforms();
    };

    const toNormalizedMouse = (clientX: number, clientY: number) => {
      const rect = canvas.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) {
        return;
      }
      mouse[0] = clamp((clientX - rect.left) / rect.width, 0, 1);
      mouse[1] = clamp(1 - (clientY - rect.top) / rect.height, 0, 1);
    };

    const onMouseMove = (event: MouseEvent) => {
      toNormalizedMouse(event.clientX, event.clientY);
    };

    const onTouchMove = (event: TouchEvent) => {
      if (!event.touches.length) {
        return;
      }
      toNormalizedMouse(event.touches[0].clientX, event.touches[0].clientY);
    };

    const onVisibilityChange = () => {
      if (document.hidden) {
        frameCount = 0;
        fpsWindowStart = 0;
      } else {
        lastTime = 0;
        frameCount = 0;
        fpsWindowStart = performance.now();
        qualityCheckStart = performance.now();
      }
    };

    let ready = false;

    const renderFrame = (now: number) => {
      if (disposed) {
        return;
      }

      rafId = window.requestAnimationFrame(renderFrame);
      if (!ready || !trailTextures || !trailFramebuffers || !noiseTexture || !logoTexture) {
        return;
      }

      const deltaTime = lastTime > 0 ? Math.min((now - lastTime) * 0.001, 0.05) : 1 / 60;
      lastTime = now;

      frameCount += 1;
      if (fpsWindowStart <= 0) {
        fpsWindowStart = now;
      }
      if (now - fpsWindowStart >= 1000) {
        const fps = Math.round((1000 * frameCount) / (now - fpsWindowStart));
        frameCount = 0;
        fpsWindowStart = now;

        if (now - qualityCheckStart >= 2000) {
          if (fps < 30 && qualityScale > minQualityScale) {
            qualityScale = Math.max(minQualityScale, qualityScale - 0.1);
            ensureTrailTargets();
          } else if (fps > 55 && qualityScale < maxQualityScale) {
            qualityScale = Math.min(maxQualityScale, qualityScale + 0.1);
            ensureTrailTargets();
          }
          qualityCheckStart = now;
        }
      }

      const mouseVelocity: [number, number] = [mouse[0] - mouseSmoothed[0], mouse[1] - mouseSmoothed[1]];
      mouseSmoothed[0] += mouseVelocity[0];
      mouseSmoothed[1] += mouseVelocity[1];

      const nextIndex = (pingPongIndex + 1) % 2;

      gl.bindFramebuffer(gl.FRAMEBUFFER, trailFramebuffers[nextIndex]);
      gl.viewport(0, 0, renderWidth, renderHeight);
      gl.useProgram(trailProgram);
      setUniform1f(trailUniforms.time, now * 0.001);
      setUniform1f(trailUniforms.deltaTime, deltaTime);
      setUniform2f(trailUniforms.mouse, mouse[0], mouse[1]);
      setUniform2f(trailUniforms.mouseVelocity, mouseVelocity[0], mouseVelocity[1]);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, noiseTexture);
      setUniform1i(trailUniforms.noiseTexture, 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, trailTextures[pingPongIndex]);
      setUniform1i(trailUniforms.previousFrame, 1);

      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, logoTexture);
      setUniform1i(trailUniforms.logoTexture, 2);

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.useProgram(renderProgram);
      setUniform1f(renderUniforms.time, now * 0.001);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, noiseTexture);
      setUniform1i(renderUniforms.noiseTexture, 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, logoTexture);
      setUniform1i(renderUniforms.logoTexture, 1);

      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, trailTextures[nextIndex]);
      setUniform1i(renderUniforms.trailTexture, 2);

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      pingPongIndex = nextIndex;
    };

    resizeCanvas();

    // Create a 1x1 transparent fallback for the logo texture
    const fallbackLogo = gl.createTexture();
    if (fallbackLogo) {
      gl.bindTexture(gl.TEXTURE_2D, fallbackLogo);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
        new Uint8Array([153, 153, 0, 0])); // neutral direction, zero mask/alpha
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      ownedTextures.push(fallbackLogo);
      logoTexture = fallbackLogo;
    }

    void loadTexture(gl, "/images/noise.png", true, maxTextureSize)
      .then((loadedNoiseTexture) => {
        if (disposed) {
          gl.deleteTexture(loadedNoiseTexture);
          return;
        }

        ownedTextures.push(loadedNoiseTexture);
        noiseTexture = loadedNoiseTexture;
        ready = true;
      })
      .catch((error) => {
        console.error(error);
      });

    window.addEventListener("mousemove", onMouseMove, { passive: true });
    window.addEventListener("touchmove", onTouchMove, { passive: true });
    window.addEventListener("resize", resizeCanvas);
    document.addEventListener("visibilitychange", onVisibilityChange);

    rafId = window.requestAnimationFrame(renderFrame);

    return () => {
      disposed = true;
      window.cancelAnimationFrame(rafId);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("touchmove", onTouchMove);
      window.removeEventListener("resize", resizeCanvas);
      document.removeEventListener("visibilitychange", onVisibilityChange);

      for (const buffer of ownedBuffers) {
        gl.deleteBuffer(buffer);
      }
      for (const texture of ownedTextures) {
        gl.deleteTexture(texture);
      }
      for (const framebuffer of ownedFramebuffers) {
        gl.deleteFramebuffer(framebuffer);
      }
      for (const program of ownedPrograms) {
        gl.deleteProgram(program);
      }
    };
  }, []);

  return (
    <div ref={containerRef} className="volumetric-light-canvas" aria-hidden="true">
      <canvas ref={canvasRef} />
    </div>
  );
}

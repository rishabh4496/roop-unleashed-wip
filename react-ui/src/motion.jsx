// ── Shared motion foundation ──────────────────────────────────────────────
// One vocabulary of spring physics + entrance variants so every animated
// surface across the app feels like the same instrument. The app's signature
// easing is cubic-bezier(0.16,1,0.3,1) ("settle without ringing"); these
// springs approximate it, with a couple of intentionally bouncier presets for
// discrete targets (buttons, toggles, thumbnails).
//
// Reduced motion is handled globally via <MotionConfig reducedMotion="user">
// in App.jsx, so transforms/layout collapse to instant when the OS asks — we
// don't need to gate every component by hand.
import React, { useRef } from 'react';
import { motion, AnimatePresence, useReducedMotion, useMotionValue, useSpring, useTransform, LayoutGroup, MotionConfig } from 'framer-motion';

export { motion, AnimatePresence, useReducedMotion, LayoutGroup, MotionConfig };

export const spring = {
  snappy: { type: 'spring', stiffness: 520, damping: 34, mass: 0.7 },
  smooth: { type: 'spring', stiffness: 260, damping: 30 },
  bouncy: { type: 'spring', stiffness: 460, damping: 20, mass: 0.8 },
  soft: { type: 'spring', stiffness: 170, damping: 22 },
};

// Cinematic entrance: rise + settle. Used by cards/panels on mount. No filter
// blur here on purpose — these panels already carry a backdrop-blur, and many
// animating a blur at once is GPU-costly. The de-blur is reserved for the
// single per-tab view transition below.
export const fadeUp = {
  hidden: { opacity: 0, y: 24, scale: 0.98 },
  show: {
    opacity: 1, y: 0, scale: 1,
    transition: { ...spring.smooth, opacity: { duration: 0.4 } },
  },
  exit: { opacity: 0, y: -12, scale: 0.985, transition: { duration: 0.2 } },
};

// Parent orchestrator: cascade children in with a staggered delay.
export const staggerParent = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06, delayChildren: 0.05 } },
  exit: { transition: { staggerChildren: 0.02, staggerDirection: -1 } },
};

// Full-view swap (tab changes): depth cross-slide.
export const viewTransition = {
  initial: { opacity: 0, y: 18, scale: 0.985, filter: 'blur(10px)' },
  animate: { opacity: 1, y: 0, scale: 1, filter: 'blur(0px)', transition: { ...spring.smooth, opacity: { duration: 0.35 }, filter: { duration: 0.35 } } },
  exit: { opacity: 0, y: -14, scale: 0.99, filter: 'blur(8px)', transition: { duration: 0.2, ease: [0.4, 0, 1, 1] } },
};

// Drop-in staggered container. Children should be <Reveal> (or any motion
// element with the fadeUp variants and no own initial/animate).
export const Stagger = ({ children, className = '', amount = 0.12, ...rest }) => (
  <motion.div
    className={className}
    variants={staggerParent}
    initial="hidden"
    animate="show"
    {...rest}
  >
    {children}
  </motion.div>
);

// A single cascading reveal element. Inherits orchestration from a <Stagger>
// parent; also works standalone (animates on mount).
export const Reveal = ({ children, className = '', as = 'div', ...rest }) => {
  const Comp = motion[as] || motion.div;
  return (
    <Comp className={className} variants={fadeUp} {...rest}>
      {children}
    </Comp>
  );
};

// Mouse-follow 3D tilt with spring physics + a soft accent glare that tracks
// the cursor. For hero / discrete cards where the depth pays off (not dense
// form cards — the movement fights inputs there).
export const TiltCard = ({ children, className = '', max = 8, glare = true, ...rest }) => {
  const ref = useRef(null);
  const reduce = useReducedMotion();
  const px = useMotionValue(0.5);
  const py = useMotionValue(0.5);
  const rx = useSpring(useTransform(py, [0, 1], [max, -max]), spring.soft);
  const ry = useSpring(useTransform(px, [0, 1], [-max, max]), spring.soft);
  const gx = useTransform(px, [0, 1], ['0%', '100%']);
  const gy = useTransform(py, [0, 1], ['0%', '100%']);
  // Computed unconditionally (hook rules) — only rendered when glare is on.
  const glareBg = useTransform([gx, gy], ([x, y]) => `radial-gradient(600px circle at ${x} ${y}, var(--accent-glow), transparent 40%)`);

  const onMove = (e) => {
    if (reduce || !ref.current) return;
    const r = ref.current.getBoundingClientRect();
    px.set((e.clientX - r.left) / r.width);
    py.set((e.clientY - r.top) / r.height);
  };
  const onLeave = () => { px.set(0.5); py.set(0.5); };

  return (
    <motion.div
      ref={ref}
      onMouseMove={onMove}
      onMouseLeave={onLeave}
      style={reduce ? undefined : { rotateX: rx, rotateY: ry, transformPerspective: 900, transformStyle: 'preserve-3d' }}
      className={`group/tilt relative ${className}`}
      {...rest}
    >
      {children}
      {glare && !reduce && (
        <motion.div
          aria-hidden
          className="pointer-events-none absolute inset-0 rounded-[inherit] opacity-0 transition-opacity duration-300 group-hover/tilt:opacity-100"
          style={{ background: glareBg }}
        />
      )}
    </motion.div>
  );
};

// Single source of truth for UI themes. `className` matches the .theme-* blocks
// in index.css (empty string = the :root default). `accent` / `bg` drive the
// swatch preview in the theme gallery; `mode` tags light vs dark.
export const THEMES = [
  { name: 'Default',         className: '',                    accent: '#E94560', bg: '#0B0C12', mode: 'dark',  label: 'Obsidian & Crimson' },
  { name: 'Midnight Violet', className: 'theme-midnight-violet', accent: '#9b5de5', bg: '#0d0819', mode: 'dark',  label: 'Deep purple glow' },
  { name: 'Sunset Amber',    className: 'theme-sunset-amber',  accent: '#f59e0b', bg: '#14100a', mode: 'dark',  label: 'Warm amber' },
  { name: 'Ocean Teal',      className: 'theme-ocean-teal',    accent: '#00bbf9', bg: '#071620', mode: 'dark',  label: 'Cyan deep-sea' },
  { name: 'Rose Gold',       className: 'theme-rose-gold',     accent: '#f472a0', bg: '#150a10', mode: 'dark',  label: 'Soft rose' },
  { name: 'Dracula',         className: 'theme-dracula',       accent: '#bd93f9', bg: '#21222c', mode: 'dark',  label: 'Classic Dracula' },
  { name: 'Solarized Dark',  className: 'theme-solarized-dark', accent: '#268bd2', bg: '#073642', mode: 'dark',  label: 'Solarized blue' },
  { name: 'Emerald Dark',    className: 'theme-emerald-dark',  accent: '#10b981', bg: '#06140d', mode: 'dark',  label: 'Forest emerald' },
  { name: 'Nordic Dark',     className: 'theme-nordic-dark',   accent: '#88c0d0', bg: '#121826', mode: 'dark',  label: 'Nord frost' },
  { name: 'Cyberpunk Dark',  className: 'theme-cyberpunk-dark', accent: '#ff007f', bg: '#12041c', mode: 'dark',  label: 'Neon cyberpunk' },
  { name: 'Monochrome',      className: 'theme-monochrome',    accent: '#d4d4d8', bg: '#0d0d0d', mode: 'dark',  label: 'Minimal grayscale' },
  { name: 'Glass Light',     className: 'theme-glass-light',   accent: '#E94560', bg: '#e5e9f0', mode: 'light', label: 'Frosted glass' },
  { name: 'Sakura Light',    className: 'theme-sakura-light',  accent: '#e0538a', bg: '#ffeef4', mode: 'light', label: 'Cherry blossom' },
  { name: 'Emerald Light',   className: 'theme-emerald-light', accent: '#059669', bg: '#e6f6e8', mode: 'light', label: 'Mint light' },
  { name: 'Nordic Light',    className: 'theme-nordic-light',  accent: '#5e81ac', bg: '#e6ebf0', mode: 'light', label: 'Nord day' },
  { name: 'Cyberpunk Light', className: 'theme-cyberpunk-light', accent: '#ff007f', bg: '#f2fff2', mode: 'light', label: 'Neon day' },

  // ── AMOLED family: pure #000000 backgrounds for OLED power saving ────────
  { name: 'AMOLED Black',    className: 'theme-amoled',         accent: '#E94560', bg: '#000000', mode: 'dark',  label: 'True black OLED' },
  { name: 'AMOLED Ice',      className: 'theme-amoled-ice',     accent: '#38bdf8', bg: '#000000', mode: 'dark',  label: 'True black · cyan' },
  { name: 'AMOLED Matrix',   className: 'theme-amoled-matrix',  accent: '#22c55e', bg: '#000000', mode: 'dark',  label: 'True black · green' },
  { name: 'AMOLED Gold',     className: 'theme-amoled-gold',    accent: '#d4af37', bg: '#000000', mode: 'dark',  label: 'True black · gold' },
  { name: 'AMOLED Violet',   className: 'theme-amoled-violet',  accent: '#a855f7', bg: '#000000', mode: 'dark',  label: 'True black · violet' },

  // ── Extra dark themes ────────────────────────────────────────────────────
  { name: 'Crimson Noir',    className: 'theme-crimson-noir',   accent: '#ef4444', bg: '#0a0505', mode: 'dark',  label: 'Dark red noir' },
  { name: 'Deep Space',      className: 'theme-deep-space',     accent: '#6366f1', bg: '#05060f', mode: 'dark',  label: 'Indigo cosmos' },
  { name: 'Gruvbox Dark',    className: 'theme-gruvbox-dark',   accent: '#fabd2f', bg: '#282828', mode: 'dark',  label: 'Retro gruvbox' },
  { name: 'Tokyo Night',     className: 'theme-tokyo-night',    accent: '#7aa2f7', bg: '#1a1b26', mode: 'dark',  label: 'Neon Tokyo' },
  { name: 'Catppuccin Mocha', className: 'theme-catppuccin-mocha', accent: '#cba6f7', bg: '#1e1e2e', mode: 'dark', label: 'Soft mocha' },
  { name: 'Blood Moon',      className: 'theme-blood-moon',     accent: '#ff2e4d', bg: '#0c0406', mode: 'dark',  label: 'Eclipse red' },
  { name: 'Forest Night',    className: 'theme-forest-night',   accent: '#4ade80', bg: '#071108', mode: 'dark',  label: 'Deep woods' },
  { name: 'Espresso',        className: 'theme-espresso',       accent: '#c8956d', bg: '#120c08', mode: 'dark',  label: 'Warm espresso' },
  { name: 'Royal Purple',    className: 'theme-royal-purple',   accent: '#a855f7', bg: '#0c0616', mode: 'dark',  label: 'Regal violet' },
  { name: 'Neon Lime',       className: 'theme-neon-lime',      accent: '#a3e635', bg: '#0a0f04', mode: 'dark',  label: 'Electric lime' },
  { name: 'Copper Rust',     className: 'theme-copper',         accent: '#e07856', bg: '#140a06', mode: 'dark',  label: 'Oxidized copper' },
  { name: 'Steel Slate',     className: 'theme-steel-slate',    accent: '#38bdf8', bg: '#0b0f14', mode: 'dark',  label: 'Cool steel' },
  { name: 'Vaporwave',       className: 'theme-vaporwave',      accent: '#ff71ce', bg: '#10061a', mode: 'dark',  label: 'Retro synthwave' },

  // ── Extra light themes ───────────────────────────────────────────────────
  { name: 'Arctic Light',    className: 'theme-arctic-light',   accent: '#0ea5e9', bg: '#eef4fa', mode: 'light', label: 'Icy blue day' },
  { name: 'Latte Light',     className: 'theme-latte-light',    accent: '#8839ef', bg: '#eff1f5', mode: 'light', label: 'Catppuccin latte' },
  { name: 'Sepia Light',     className: 'theme-sepia-light',    accent: '#b45309', bg: '#f4ece0', mode: 'light', label: 'Warm paper' },
];

export const THEME_CLASSES = THEMES.map((t) => t.className).filter(Boolean);

export const themeByName = (name) => THEMES.find((t) => t.name === name) || THEMES[0];

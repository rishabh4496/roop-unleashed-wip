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
];

export const THEME_CLASSES = THEMES.map((t) => t.className).filter(Boolean);

export const themeByName = (name) => THEMES.find((t) => t.name === name) || THEMES[0];

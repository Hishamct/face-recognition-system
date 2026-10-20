/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        dark: {
          950: '#090D16', // Deep charcoal/obsidian
          900: '#0F172A', // Slate 900
          850: '#151F32', // Elevated card surface
          800: '#1E293B', // Slate 800
          700: '#334155', // Slate 700
          600: '#475569',
        },
        brand: {
          50: '#EEF2FF',
          100: '#E0E7FF',
          500: '#6366F1', // Indigo
          600: '#4F46E5', // Primary brand indigo
          700: '#4338CA',
        },
        accent: {
          blue: '#0284C7', // Refined sky blue (not neon cyan)
          emerald: '#059669', // Subtle emerald
          amber: '#D97706', // Soft amber
          rose: '#E11D48', // Soft rose
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
}

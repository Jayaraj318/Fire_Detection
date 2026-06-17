/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        './pages/**/*.{js,ts,jsx,tsx,mdx}',
        './components/**/*.{js,ts,jsx,tsx,mdx}',
        './app/**/*.{js,ts,jsx,tsx,mdx}',
    ],
    theme: {
        extend: {
            colors: {
                fire: {
                    50: '#fef3f2',
                    100: '#fee4e2',
                    200: '#fecdca',
                    300: '#fca5a1',
                    400: '#f87168',
                    500: '#ef4444',
                    600: '#dc2626',
                    700: '#b91c1c',
                    800: '#991b1b',
                    900: '#7f1d1d',
                },
            },
            animation: {
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                'fire-glow': 'fire-glow 1.5s ease-in-out infinite alternate',
            },
            keyframes: {
                'fire-glow': {
                    '0%': { boxShadow: '0 0 20px 10px rgba(239, 68, 68, 0.3)' },
                    '100%': { boxShadow: '0 0 30px 15px rgba(239, 68, 68, 0.6)' },
                },
            },
        },
    },
    plugins: [],
};

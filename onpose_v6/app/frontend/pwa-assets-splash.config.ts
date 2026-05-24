import { createAppleSplashScreens, defineConfig } from "@vite-pwa/assets-generator/config"

export default defineConfig({
  preset: {
    transparent: { sizes: [], favicons: [] },
    maskable: { sizes: [] },
    apple: { sizes: [] },
    appleSplashScreens: createAppleSplashScreens(
      {
        padding: 0,
        resizeOptions: { background: "#4F46E5", fit: "contain" },
        linkMediaOptions: { log: true },
      },
      [
        "iPhone 16 Pro Max",
        "iPhone 16",
        "iPhone 15 Pro Max",
        "iPhone 14",
        "iPhone 13 mini",
        'iPhone SE 4.7"',
        'iPad Pro 12.9"',
        'iPad mini 8.3"',
      ],
    ),
  },
  images: ["public/splash-source.svg"],
})

import { defineConfig, minimal2023Preset } from "@vite-pwa/assets-generator/config"

export default defineConfig({
  preset: {
    ...minimal2023Preset,
    maskable: {
      ...minimal2023Preset.maskable,
      resizeOptions: {
        ...minimal2023Preset.maskable.resizeOptions,
        background: "#FFFFFF",
      },
    },
    apple: {
      ...minimal2023Preset.apple,
      padding: 0.05,
      resizeOptions: {
        ...minimal2023Preset.apple.resizeOptions,
        background: "#FFFFFF",
      },
    },
  },
  images: ["public/favicon.svg"],
})

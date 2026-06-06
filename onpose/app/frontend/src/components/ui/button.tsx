import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Slot } from "radix-ui"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "group/button inline-flex shrink-0 items-center justify-center rounded-full border border-transparent bg-clip-padding text-sm font-bold whitespace-nowrap transition-all outline-none select-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/40 active:not-aria-[haspopup]:translate-y-px active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50 aria-invalid:border-destructive aria-invalid:ring-3 aria-invalid:ring-destructive/20 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
  {
    variants: {
      variant: {
        // 보라 배경 위 1차 CTA = 라임 알약
        default:
          "bg-[#E6FB4D] text-[#15110A] shadow-[0_10px_24px_-10px_rgba(0,0,0,0.45)] hover:brightness-95 [a]:hover:brightness-95",
        // 흰 카드 위/보라 위 모두 통하는 흰 알약 (테두리로 흰 배경 위에서도 보임)
        outline:
          "border-[#E5E8EB] bg-white text-[#191F28] shadow-[0_8px_20px_-12px_rgba(20,16,60,0.4)] hover:bg-[#F4F4FF] aria-expanded:bg-[#F4F4FF]",
        // 연보라 알약
        secondary:
          "bg-[#EAEAFB] text-[#4E45E6] hover:brightness-95 aria-expanded:bg-[#EAEAFB]",
        ghost:
          "text-white/90 hover:bg-white/12 hover:text-white aria-expanded:bg-white/12",
        destructive:
          "bg-[#FF5A1F] text-white hover:brightness-95 focus-visible:ring-[#FF5A1F]/30",
        link: "rounded-none text-[#E6FB4D] underline-offset-4 hover:underline",
      },
      size: {
        default:
          "h-12 gap-1.5 px-5 has-data-[icon=inline-end]:pr-4 has-data-[icon=inline-start]:pl-4",
        xs: "h-7 gap-1 px-3 text-xs has-data-[icon=inline-end]:pr-2 has-data-[icon=inline-start]:pl-2 [&_svg:not([class*='size-'])]:size-3",
        sm: "h-9 gap-1 px-4 text-[0.8rem] has-data-[icon=inline-end]:pr-3 has-data-[icon=inline-start]:pl-3 [&_svg:not([class*='size-'])]:size-3.5",
        lg: "h-14 gap-2 px-6 text-[1rem] has-data-[icon=inline-end]:pr-5 has-data-[icon=inline-start]:pl-5",
        icon: "size-12",
        "icon-xs": "size-7 [&_svg:not([class*='size-'])]:size-3",
        "icon-sm": "size-9",
        "icon-lg": "size-12",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Button({
  className,
  variant = "default",
  size = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot.Root : "button"

  return (
    <Comp
      data-slot="button"
      data-variant={variant}
      data-size={size}
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }

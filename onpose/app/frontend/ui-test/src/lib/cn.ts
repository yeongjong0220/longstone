// 본 프로젝트의 @/lib/utils cn(clsx+tailwind-merge) 패턴을 미러링한 경량 버전.
// 테스트 격리를 위해 외부 의존 없이 구현.
export function cn(
  ...classes: Array<string | false | null | undefined>
): string {
  return classes.filter(Boolean).join(" ")
}

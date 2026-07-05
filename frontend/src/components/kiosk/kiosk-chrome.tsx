import * as React from "react";

/** Corner brackets around the tracked face. Thick and ink-colored — industrial
 *  brutalist, not HUD. Rendered inside a positioned box. Shared by attendance +
 *  enrollment. `border-foreground` = white here (the kiosk is dark-locked). */
export function Brackets({ size = 28 }: { size?: number }) {
  const c = "absolute border-foreground";
  const s = { width: size, height: size };
  return (
    <>
      <span className={`${c} left-0 top-0 border-l-4 border-t-4`} style={s} />
      <span className={`${c} right-0 top-0 border-r-4 border-t-4`} style={s} />
      <span className={`${c} bottom-0 left-0 border-b-4 border-l-4`} style={s} />
      <span className={`${c} bottom-0 right-0 border-b-4 border-r-4`} style={s} />
    </>
  );
}

/** Full-screen overlay for booting / error / result states. Dim backdrop over
 *  the camera, content in a hard-shadowed brutalist card so every kiosk state
 *  reads with the same poster language as the admin pages. */
export function Overlay({ children }: { children: React.ReactNode }) {
  return (
    <div className="absolute inset-0 z-30 flex items-center justify-center bg-background/95 px-6 animate-in fade-in duration-300">
      <div className="flex max-w-md flex-col items-center gap-6 rounded-[3px] border-2 border-foreground bg-card px-10 py-12 text-center shadow-brutal-lg">
        {children}
      </div>
    </div>
  );
}

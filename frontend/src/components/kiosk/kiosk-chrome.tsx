import * as React from "react";

/** Corner brackets around the tracked face. Square and white — industrial,
 *  not HUD. Rendered inside a positioned box. Shared by attendance + enrollment. */
export function Brackets({ size = 28 }: { size?: number }) {
  const c = "absolute border-white";
  const s = { width: size, height: size };
  return (
    <>
      <span className={`${c} left-0 top-0 border-l-[3px] border-t-[3px]`} style={s} />
      <span className={`${c} right-0 top-0 border-r-[3px] border-t-[3px]`} style={s} />
      <span className={`${c} bottom-0 left-0 border-b-[3px] border-l-[3px]`} style={s} />
      <span className={`${c} bottom-0 right-0 border-b-[3px] border-r-[3px]`} style={s} />
    </>
  );
}

/** Full-screen solid overlay for booting / error / result states. */
export function Overlay({ children }: { children: React.ReactNode }) {
  return (
    <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-6 bg-zinc-950 px-6 text-center animate-in fade-in duration-300">
      {children}
    </div>
  );
}

import type { Metadata } from "next";
import { KioskScreen } from "@/components/kiosk/kiosk-screen";

export const metadata: Metadata = {
  title: "Kiosk điểm danh",
  description: "Màn hình điểm danh khuôn mặt tại chỗ.",
};

export default function KioskPage() {
  return <KioskScreen />;
}

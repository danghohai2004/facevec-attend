import type { Metadata } from "next";
import { KioskEnrollment } from "@/components/kiosk/kiosk-enrollment";
import { KioskScreen } from "@/components/kiosk/kiosk-screen";

export const metadata: Metadata = {
  title: "Kiosk điểm danh",
  description: "Màn hình điểm danh khuôn mặt tại chỗ.",
};

export default async function KioskPage({
  searchParams,
}: {
  searchParams: Promise<{
    [key: string]: string | string[] | undefined;
  }>;
}) {
  const params = await searchParams;
  const first = (value: string | string[] | undefined) =>
    Array.isArray(value) ? value[0] : value;

  if (first(params.mode) === "register") {
    return (
      <KioskEnrollment
        name={first(params.name) ?? ""}
        empCode={first(params.emp_code) ?? ""}
      />
    );
  }

  return <KioskScreen />;
}

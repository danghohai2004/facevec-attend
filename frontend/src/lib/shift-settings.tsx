"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { z } from "zod";
import { toast } from "sonner";
import { getShiftSettings, updateShiftSettings } from "@/lib/api";
import type { ShiftSettings } from "@/lib/types";

const STORAGE_KEY = "facevec.shift-settings";

export const ShiftSettingsSchema = z.object({
  checkInStart: z.string().min(1, "Check-in start time is required."),
  checkInEnd: z.string().min(1, "Check-in end time is required."),
  checkOutStart: z.string().min(1, "Check-out start time is required."),
  checkOutEnd: z.string().min(1, "Check-out end time is required."),
});

export const DEFAULT_SHIFT_SETTINGS: ShiftSettings = {
  checkInStart: "08:00",
  checkInEnd: "10:00",
  checkOutStart: "17:00",
  checkOutEnd: "19:00",
};

type ShiftSettingsContextValue = {
  settings: ShiftSettings;
  updateSettings: (next: ShiftSettings) => Promise<boolean>;
};

const ShiftSettingsContext = createContext<ShiftSettingsContextValue | undefined>(
  undefined,
);

export function ShiftSettingsProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [settings, setSettings] =
    useState<ShiftSettings>(DEFAULT_SHIFT_SETTINGS);
  const hasHydratedRef = useRef(false);

  useEffect(() => {
    let isMounted = true;

    const loadFromStorage = () => {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return;
      }
      let parsedValue: unknown;
      try {
        parsedValue = JSON.parse(raw);
      } catch {
        toast.error("Saved shift settings could not be read.");
        return;
      }

      const parsed = ShiftSettingsSchema.safeParse(parsedValue);
      if (parsed.success) {
        setSettings(parsed.data);
      } else {
        toast.error("Saved shift settings are invalid. Defaults restored.");
      }
    };

    const loadFromApi = async () => {
      try {
        const apiSettings = await getShiftSettings();
        if (isMounted) {
          setSettings(apiSettings);
        }
      } catch {
        toast.error("Could not load shift settings from server.");
      }
    };

    loadFromStorage();
    void loadFromApi();

    return () => {
      isMounted = false;
    };
  }, []);

  const persistSettings = useCallback(async (next: ShiftSettings) => {
    try {
      const saved = await updateShiftSettings(next);
      setSettings(saved);
      return true;
    } catch {
      toast.error("Could not save shift settings.");
      return false;
    }
  }, []);

  useEffect(() => {
    if (!hasHydratedRef.current) {
      hasHydratedRef.current = true;
      return;
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  }, [settings]);

  const value = useMemo(
    () => ({
      settings,
      updateSettings: persistSettings,
    }),
    [persistSettings, settings],
  );

  return (
    <ShiftSettingsContext.Provider value={value}>
      {children}
    </ShiftSettingsContext.Provider>
  );
}

export function useShiftSettings() {
  const context = useContext(ShiftSettingsContext);
  if (!context) {
    throw new Error("useShiftSettings must be used within ShiftSettingsProvider");
  }
  return context;
}

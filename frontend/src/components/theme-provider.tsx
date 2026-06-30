"use client";

import * as React from "react";

type ThemeValue = "light" | "dark" | "system";
type DataAttribute = `data-${string}`;
type ThemeAttribute = "class" | DataAttribute;

export type ThemeProviderProps = React.PropsWithChildren<{
  themes?: ThemeValue[];
  forcedTheme?: ThemeValue;
  enableSystem?: boolean;
  enableColorScheme?: boolean;
  storageKey?: string;
  defaultTheme?: ThemeValue;
  attribute?: ThemeAttribute | ThemeAttribute[];
  value?: Partial<Record<ThemeValue, string>>;
}>;

type ThemeContextValue = {
  theme?: ThemeValue;
  setTheme: React.Dispatch<React.SetStateAction<ThemeValue>>;
  forcedTheme?: ThemeValue;
  resolvedTheme?: "light" | "dark";
  systemTheme?: "light" | "dark";
  themes: ThemeValue[];
};

const ThemeContext = React.createContext<ThemeContextValue | undefined>(undefined);

const MEDIA_QUERY = "(prefers-color-scheme: dark)";
const DEFAULT_THEMES: ThemeValue[] = ["light", "dark"];

function getSystemTheme() {
  if (typeof window === "undefined") {
    return undefined;
  }
  return window.matchMedia(MEDIA_QUERY).matches ? "dark" : "light";
}

function getThemeValue(
  theme: ThemeValue,
  enableSystem: boolean,
  resolvedTheme: "light" | "dark" | undefined,
) {
  if (theme === "system" && enableSystem) {
    return resolvedTheme;
  }
  return theme === "system" ? undefined : theme;
}

function applyAttribute(
  attribute: ThemeAttribute,
  value: string | undefined,
  availableValues: string[],
) {
  const root = document.documentElement;
  if (attribute === "class") {
    root.classList.remove(...availableValues);
    if (value) {
      root.classList.add(value);
    }
    return;
  }

  if (value) {
    root.setAttribute(attribute, value);
  } else {
    root.removeAttribute(attribute);
  }
}

export function ThemeProvider({
  children,
  attribute = "data-theme",
  defaultTheme = "system",
  enableSystem = true,
  enableColorScheme = true,
  storageKey = "theme",
  themes = DEFAULT_THEMES,
  value,
  forcedTheme,
}: ThemeProviderProps) {
  const [theme, setThemeState] = React.useState<ThemeValue>(() => {
    if (typeof window === "undefined") {
      return defaultTheme;
    }
    const storedTheme = window.localStorage.getItem(storageKey) as ThemeValue | null;
    return storedTheme ?? defaultTheme;
  });
  const [resolvedTheme, setResolvedTheme] = React.useState<"light" | "dark" | undefined>(
    getSystemTheme(),
  );

  const availableValues = React.useMemo(() => {
    if (value) {
      return Object.values(value);
    }
    return themes;
  }, [themes, value]);

  const setTheme = React.useCallback(
    (next: React.SetStateAction<ThemeValue>) => {
      setThemeState((prev) => {
        const resolved = typeof next === "function" ? next(prev) : next;
        if (typeof window !== "undefined") {
          window.localStorage.setItem(storageKey, resolved);
        }
        return resolved;
      });
    },
    [storageKey],
  );

  React.useEffect(() => {
    if (!enableSystem || typeof window === "undefined") {
      return;
    }
    const media = window.matchMedia(MEDIA_QUERY);
    const handleChange = () => {
      setResolvedTheme(media.matches ? "dark" : "light");
    };
    handleChange();
    media.addEventListener?.("change", handleChange);
    media.addListener?.(handleChange);
    return () => {
      media.removeEventListener?.("change", handleChange);
      media.removeListener?.(handleChange);
    };
  }, [enableSystem]);

  React.useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }
    const activeTheme = forcedTheme ?? theme;
    const appliedTheme = getThemeValue(activeTheme, enableSystem, resolvedTheme);
    const attributes = Array.isArray(attribute) ? attribute : [attribute];
    const mappedValue = appliedTheme && value ? value[appliedTheme] : appliedTheme;

    attributes.forEach((attr) => applyAttribute(attr, mappedValue, availableValues));

    if (enableColorScheme) {
      document.documentElement.style.colorScheme =
        appliedTheme === "dark" || appliedTheme === "light" ? appliedTheme : "";
    }
  }, [
    attribute,
    availableValues,
    enableColorScheme,
    enableSystem,
    forcedTheme,
    resolvedTheme,
    theme,
    value,
  ]);

  const contextValue = React.useMemo<ThemeContextValue>(
    () => ({
      theme,
      setTheme,
      forcedTheme,
      resolvedTheme: theme === "system" ? resolvedTheme : theme,
      systemTheme: enableSystem ? resolvedTheme : undefined,
      themes: enableSystem ? [...themes, "system"] : themes,
    }),
    [enableSystem, forcedTheme, resolvedTheme, setTheme, theme, themes],
  );

  return <ThemeContext.Provider value={contextValue}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  return (
    React.useContext(ThemeContext) ?? {
      setTheme: () => {},
      themes: [],
    }
  );
}

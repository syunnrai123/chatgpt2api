"use client";

import { useEffect, useState } from "react";

import { fetchAppText } from "@/lib/api";
import { DEFAULT_APP_TEXT, normalizeAppText, type AppTextSettings } from "@/lib/app-text";

export function useAppText(enabled = true): AppTextSettings {
  const [appText, setAppText] = useState<AppTextSettings>(DEFAULT_APP_TEXT);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    let active = true;
    void fetchAppText()
      .then((data) => {
        if (active) {
          setAppText(normalizeAppText(data.app_text));
        }
      })
      .catch(() => {
        if (active) {
          setAppText(DEFAULT_APP_TEXT);
        }
      });

    return () => {
      active = false;
    };
  }, [enabled]);

  return appText;
}

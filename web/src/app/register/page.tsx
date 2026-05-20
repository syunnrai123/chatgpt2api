"use client";

import { useEffect, useRef } from "react";
import { LoaderCircle } from "lucide-react";
import { toast } from "sonner";

import webConfig from "@/constants/common-env";
import { useAuthGuard } from "@/lib/use-auth-guard";
import type { RegisterConfig } from "@/lib/api";
import { getStoredAuthKey } from "@/store/auth";

import { useSettingsStore } from "../settings/store";
import { RegisterCard } from "./components/register-card";

function RegisterDataController() {
  const didLoadRef = useRef(false);
  const loadRegister = useSettingsStore((state) => state.loadRegister);
  const setRegisterConfig = useSettingsStore((state) => state.setRegisterConfig);

  useEffect(() => {
    if (didLoadRef.current) return;
    didLoadRef.current = true;
    void loadRegister();
  }, [loadRegister]);

  useEffect(() => {
    let source: EventSource | null = null;
    let closed = false;
    let reportedError = false;
    let lastFallbackAt = 0;
    const loadFallback = () => {
      const now = Date.now();
      if (now - lastFallbackAt >= 5000) {
        lastFallbackAt = now;
        void loadRegister(true);
      }
    };
    void getStoredAuthKey().then((token) => {
      if (closed || !token) return;
      const baseUrl = webConfig.apiUrl.replace(/\/$/, "");
      source = new EventSource(`${baseUrl}/api/register/events?token=${encodeURIComponent(token)}`);
      source.onmessage = (event) => {
        try {
          setRegisterConfig(JSON.parse(event.data) as RegisterConfig);
        } catch {
          if (!reportedError) {
            reportedError = true;
            toast.error("注册机实时数据解析失败，已重新拉取最新配置");
          }
          loadFallback();
        }
      };
      source.onerror = () => {
        if (closed) return;
        if (!reportedError) {
          reportedError = true;
          toast.error("注册机实时连接异常，正在等待自动重连，并已重新拉取最新配置");
        }
        loadFallback();
      };
    });
    return () => {
      closed = true;
      source?.close();
    };
  }, [loadRegister, setRegisterConfig]);

  return null;
}

function RegisterPageContent() {
  return (
    <>
      <RegisterDataController />
      <section className="mb-2 flex flex-col gap-1 lg:flex-row lg:items-start lg:justify-between">
        <div className="space-y-1">
          <div className="text-xs font-semibold tracking-[0.18em] text-stone-500 uppercase">Register</div>
          <h1 className="text-2xl font-semibold tracking-tight">ChatGPT注册机</h1>
        </div>
      </section>
      <section>
        <RegisterCard />
      </section>
    </>
  );
}

export default function RegisterPage() {
  const { isCheckingAuth, session } = useAuthGuard(["admin"]);

  if (isCheckingAuth || !session || session.role !== "admin") {
    return (
      <div className="flex min-h-[40vh] items-center justify-center">
        <LoaderCircle className="size-5 animate-spin text-stone-400" />
      </div>
    );
  }

  return <RegisterPageContent />;
}

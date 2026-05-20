export type AppTextSettings = {
  brand_name: string;
  github_label: string;
  github_url: string;
  register_eyebrow: string;
  register_title: string;
};

export const DEFAULT_APP_TEXT: AppTextSettings = {
  brand_name: "chatgpt2api",
  github_label: "GitHub",
  github_url: "https://github.com/basketikun/chatgpt2api",
  register_eyebrow: "Register",
  register_title: "ChatGPT注册机",
};

function normalizeHttpUrl(value: unknown, fallback: string): string {
  const raw = String(value || "").trim();
  if (!raw) {
    return fallback;
  }

  try {
    const url = new URL(raw);
    return url.protocol === "http:" || url.protocol === "https:" ? raw : fallback;
  } catch {
    return fallback;
  }
}

export function normalizeAppText(value: unknown): AppTextSettings {
  const source = value && typeof value === "object" ? (value as Partial<Record<keyof AppTextSettings, unknown>>) : {};

  return {
    brand_name: String(source.brand_name || DEFAULT_APP_TEXT.brand_name).trim() || DEFAULT_APP_TEXT.brand_name,
    github_label: String(source.github_label || DEFAULT_APP_TEXT.github_label).trim() || DEFAULT_APP_TEXT.github_label,
    github_url: normalizeHttpUrl(source.github_url, DEFAULT_APP_TEXT.github_url),
    register_eyebrow:
      String(source.register_eyebrow || DEFAULT_APP_TEXT.register_eyebrow).trim() ||
      DEFAULT_APP_TEXT.register_eyebrow,
    register_title:
      String(source.register_title || DEFAULT_APP_TEXT.register_title).trim() || DEFAULT_APP_TEXT.register_title,
  };
}

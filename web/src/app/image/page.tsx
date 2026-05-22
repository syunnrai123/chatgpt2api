"use client";

import {useCallback, useEffect, useMemo, useRef, useState} from "react";
import {History, LoaderCircle, Plus, Trash2} from "lucide-react";
import {toast} from "sonner";

import {ImageComposer} from "@/app/image/components/image-composer";
import {ImageResults, type ImageLightboxItem} from "@/app/image/components/image-results";
import {ImageSidebar} from "@/app/image/components/image-sidebar";
import {ImageLightbox} from "@/components/image-lightbox";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import {Button} from "@/components/ui/button";
import {
    createImageEditTask,
    createImageGenerationTask,
    fetchAccounts,
    fetchCurrentIdentity,
    fetchImageTasks,
    type Account,
    type ImageResolution,
    type ImageTask,
} from "@/lib/api";
import {useAuthGuard} from "@/lib/use-auth-guard";
import {
    clearImageConversations,
    deleteImageConversation,
    getImageConversationStats,
    listImageConversations,
    renameImageConversation,
    saveImageConversation,
    saveImageConversations,
    type ImageConversation,
    type ImageConversationMode,
    type ImageTurn,
    type StoredImage,
    type StoredReferenceImage,
} from "@/store/image-conversations";

const ACTIVE_CONVERSATION_STORAGE_KEY = "chatgpt2api:image_active_conversation_id";
const IMAGE_SIZE_STORAGE_KEY = "chatgpt2api:image_last_size";
const IMAGE_RESOLUTION_STORAGE_KEY = "chatgpt2api:image_last_resolution";
const IMAGE_COUNT_STORAGE_KEY = "chatgpt2api:image_last_count";
const CONTINUOUS_REFERENCE_STORAGE_KEY = "chatgpt2api:image_continuous_reference";
const SUBMIT_IMAGE_TASK_CONCURRENCY = 4;
const DEFAULT_MAX_IMAGES_PER_TASK = 20;
const IMAGE_CONTEXT_MAX_TURNS = 8;
const IMAGE_CONTEXT_MAX_CHARS = 3600;
const IMAGE_RESOLUTION_ORDER: ImageResolution[] = ["1k", "2k", "4k"];
const RESOLUTION_FALLBACK_ERROR_PATTERNS = [
    /resolution/i,
    /high[-\s]?res/i,
    /pixel/i,
    /dimension/i,
    /image size/i,
    /requested size/i,
    /maximum.*resolution/i,
    /4k/i,
    /2k/i,
    /分辨率/,
    /清晰度/,
    /像素/,
    /尺寸/,
];
const NON_RESOLUTION_FALLBACK_ERROR_PATTERNS = [
    /content.*policy/i,
    /policy.*violation/i,
    /safety/i,
    /quota/i,
    /rate.?limit/i,
    /queue/i,
    /auth/i,
    /token/i,
    /账号/,
    /额度/,
    /限流/,
    /队列/,
    /认证/,
    /封禁/,
    /敏感/,
    /安全/,
];

function normalizeImageResolution(value: unknown): ImageResolution {
    const text = String(value || "").trim().toLowerCase();
    return text === "2k" || text === "4k" ? text : "1k";
}

function imageResolutionLabel(value: ImageResolution) {
    return value.toUpperCase();
}

function nextLowerImageResolution(value: unknown): ImageResolution | null {
    const index = IMAGE_RESOLUTION_ORDER.indexOf(normalizeImageResolution(value));
    return index > 0 ? IMAGE_RESOLUTION_ORDER[index - 1] : null;
}

function isResolutionFallbackError(message: unknown) {
    const text = String(message || "").trim();
    if (!text) {
        return false;
    }
    return (
        RESOLUTION_FALLBACK_ERROR_PATTERNS.some((pattern) => pattern.test(text)) &&
        !NON_RESOLUTION_FALLBACK_ERROR_PATTERNS.some((pattern) => pattern.test(text))
    );
}

function fallbackTaskId(imageId: string, resolution: ImageResolution) {
    return `${imageId}-${resolution}-${createId()}`;
}

function normalizeMaxImageCount(value: unknown) {
    const parsed = Math.floor(Number(value));
    return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_MAX_IMAGES_PER_TASK;
}

function clampImageCount(value: string, maxCount = DEFAULT_MAX_IMAGES_PER_TASK) {
    return String(Math.min(maxCount, Math.max(1, Math.floor(Number(value) || 1))));
}

function compactContextText(value: unknown, maxLength = 360) {
    const text = String(value || "").replace(/\s+/g, " ").trim();
    return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

function successfulImageCount(turn: ImageTurn) {
    return turn.resultsDeleted ? 0 : turn.images.filter((image) => image.status === "success").length;
}

function revisedPromptSummary(turn: ImageTurn) {
    if (turn.resultsDeleted) {
        return "";
    }
    const prompts = turn.images
        .map((image) => compactContextText(image.revised_prompt, 220))
        .filter(Boolean);
    return Array.from(new Set(prompts)).slice(0, 2).join("；");
}

function buildTurnContextLine(turn: ImageTurn, turnNumber: number) {
    const prompt = compactContextText(turn.prompt);
    if (!prompt) {
        return "";
    }
    const parts = [
        `第 ${turnNumber} 轮`,
        turn.mode === "edit" ? "图生图" : "文生图",
        `用户请求：${prompt}`,
    ];
    if (turn.mode === "edit" && turn.referenceImages.length > 0) {
        parts.push(`参考图 ${turn.referenceImages.length} 张`);
    }
    const successCount = successfulImageCount(turn);
    if (successCount > 0) {
        parts.push(`已成功生成 ${successCount} 张`);
    }
    const revisedPrompt = revisedPromptSummary(turn);
    if (revisedPrompt) {
        parts.push(`结果修订提示：${revisedPrompt}`);
    }
    return parts.join("；");
}

function buildContextualImagePrompt(conversation: ImageConversation, activeTurn: ImageTurn) {
    const activeIndex = conversation.turns.findIndex((turn) => turn.id === activeTurn.id);
    if (activeIndex <= 0) {
        return activeTurn.prompt;
    }
    const historyLines = conversation.turns
        .slice(0, activeIndex)
        .map((turn, index) => ({turn, turnNumber: index + 1}))
        .filter(({turn}) => !turn.promptDeleted && turn.prompt.trim())
        .slice(-IMAGE_CONTEXT_MAX_TURNS)
        .map(({turn, turnNumber}) => buildTurnContextLine(turn, turnNumber))
        .filter(Boolean);
    if (historyLines.length === 0) {
        return activeTurn.prompt;
    }
    const historyText = historyLines.join("\n");
    const compactHistory =
        historyText.length > IMAGE_CONTEXT_MAX_CHARS
            ? `...${historyText.slice(historyText.length - IMAGE_CONTEXT_MAX_CHARS)}`
            : historyText;
    return [
        "你正在同一个图片会话中继续创作。请结合历史上下文，保持已经确定的主体、风格、场景和约束一致；如果当前请求与历史上下文冲突，以当前请求为准。",
        "",
        "历史上下文：",
        compactHistory,
        "",
        "当前请求：",
        activeTurn.prompt,
    ].join("\n");
}

function ensureEditableReferences(turn: ImageTurn, referenceFiles: File[]) {
    if (turn.mode === "edit" && referenceFiles.length === 0) {
        throw new Error("未找到可用于继续编辑的参考图");
    }
}

const activeConversationQueueIds = new Set<string>();

function buildConversationTitle(prompt: string) {
    const trimmed = prompt.trim();
    if (trimmed.length <= 12) {
        return trimmed;
    }
    return `${trimmed.slice(0, 12)}...`;
}

function formatConversationTime(value: string) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return "";
    }
    return new Intl.DateTimeFormat("zh-CN", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
    }).format(date);
}

function formatAvailableQuota(accounts: Account[]) {
    const availableAccounts = accounts.filter((account) => account.status !== "禁用");
    return String(availableAccounts.reduce((sum, account) => sum + Math.max(0, account.quota), 0));
}

function formatUserQuota(value?: number | null) {
    const amount = Number(value ?? 0);
    if (!Number.isFinite(amount)) {
        return "0";
    }
    return new Intl.NumberFormat("zh-CN", {maximumFractionDigits: 6}).format(amount);
}

function createId() {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
        return crypto.randomUUID();
    }
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function readFileAsDataUrl(file: File) {
    return new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(new Error("读取参考图失败"));
        reader.readAsDataURL(file);
    });
}

function dataUrlToFile(dataUrl: string, fileName: string, mimeType?: string) {
    const [header, content] = dataUrl.split(",", 2);
    const matchedMimeType = header.match(/data:(.*?);base64/)?.[1];
    const binary = atob(content || "");
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
    }
    return new File([bytes], fileName, {type: mimeType || matchedMimeType || "image/png"});
}

function buildReferenceImageFromResult(image: StoredImage, fileName: string): StoredReferenceImage | null {
    if (!image.b64_json) {
        return null;
    }

    return {
        name: fileName,
        type: "image/png",
        dataUrl: `data:image/png;base64,${image.b64_json}`,
    };
}

async function fetchImageAsFile(url: string, fileName: string) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error("读取结果图失败");
    }
    const blob = await response.blob();
    return new File([blob], fileName, {type: blob.type || "image/png"});
}

async function buildReferenceImageFromStoredImage(image: StoredImage, fileName: string) {
    const direct = buildReferenceImageFromResult(image, fileName);
    if (direct) {
        return {
            referenceImage: direct,
            file: dataUrlToFile(direct.dataUrl, direct.name, direct.type),
        };
    }

    if (!image.url) {
        return null;
    }
    const file = await fetchImageAsFile(image.url, fileName);
    return {
        referenceImage: {
            name: file.name,
            type: file.type || "image/png",
            dataUrl: await readFileAsDataUrl(file),
        },
        file,
    };
}

async function buildAutomaticReferenceImages(conversation: ImageConversation | null, beforeTurnId?: string) {
    if (!conversation || conversation.turns.length === 0) {
        return {referenceImages: [] as StoredReferenceImage[], referenceFiles: [] as File[]};
    }

    const endIndex = beforeTurnId
        ? conversation.turns.findIndex((turn) => turn.id === beforeTurnId)
        : conversation.turns.length;
    const safeEndIndex = endIndex >= 0 ? endIndex : conversation.turns.length;

    for (let index = safeEndIndex - 1; index >= 0; index -= 1) {
        const turn = conversation.turns[index];
        if (turn.resultsDeleted) {
            continue;
        }
        const sourceImage = turn.images.find((image) => image.status === "success" && (image.b64_json || image.url));
        if (!sourceImage) {
            continue;
        }
        try {
            const built = await buildReferenceImageFromStoredImage(sourceImage, `${turn.id}-auto-reference.png`);
            if (!built) {
                continue;
            }
            return {
                referenceImages: [built.referenceImage],
                referenceFiles: [built.file],
            };
        } catch {
            continue;
        }
    }

    return {referenceImages: [] as StoredReferenceImage[], referenceFiles: [] as File[]};
}

function taskDataToStoredImage(image: StoredImage, task: ImageTask): StoredImage {
    if (task.status === "success") {
        const first = task.data?.[0];
        if (!first?.b64_json && !first?.url) {
            return {
                ...image,
                taskId: task.id,
                status: "error",
                error: task.data_expired ? task.error || "图片未保存到服务器，结果已过期，请重新生成" : "未返回图片数据",
            };
        }
        return {
            ...image,
            taskId: task.id,
            status: "success",
            b64_json: first.b64_json,
            url: first.url,
            revised_prompt: first.revised_prompt,
            error: undefined,
        };
    }

    if (task.status === "error") {
        return {
            ...image,
            taskId: task.id,
            status: "error",
            error: task.error || "生成失败",
        };
    }

    return {
        ...image,
        taskId: task.id,
        status: "loading",
        error: undefined,
    };
}

function sleep(ms: number) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function forEachWithConcurrency<T>(
    items: T[],
    concurrency: number,
    worker: (item: T) => Promise<void>,
) {
    let nextIndex = 0;
    const workerCount = Math.min(Math.max(1, concurrency), items.length);
    await Promise.all(
        Array.from({length: workerCount}, async () => {
            while (nextIndex < items.length) {
                const item = items[nextIndex];
                nextIndex += 1;
                await worker(item);
            }
        }),
    );
}

function pickFallbackConversationId(conversations: ImageConversation[]) {
    const activeConversation = conversations.find((conversation) =>
        conversation.turns.some((turn) => turn.status === "queued" || turn.status === "generating"),
    );
    return activeConversation?.id ?? conversations[0]?.id ?? null;
}

function sortImageConversations(conversations: ImageConversation[]) {
    return [...conversations].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
}

function deriveTurnStatus(turn: ImageTurn): Pick<ImageTurn, "status" | "error"> {
    const loadingCount = turn.images.filter((image) => image.status === "loading").length;
    const failedCount = turn.images.filter((image) => image.status === "error").length;
    const successCount = turn.images.filter((image) => image.status === "success").length;
    if (loadingCount > 0) {
        return {status: turn.status === "queued" ? "queued" : "generating", error: undefined};
    }
    if (failedCount > 0) {
        return {status: "error", error: `其中 ${failedCount} 张未成功生成`};
    }
    if (successCount > 0) {
        return {status: "success", error: undefined};
    }
    return {status: "queued", error: undefined};
}

async function syncConversationImageTasks(items: ImageConversation[]) {
    const taskIds = Array.from(
        new Set(
            items.flatMap((conversation) =>
                conversation.turns.flatMap((turn) =>
                    turn.resultsDeleted
                        ? []
                        : turn.images.flatMap((image) => (image.status === "loading" && image.taskId ? [image.taskId] : [])),
                ),
            ),
        ),
    );
    if (taskIds.length === 0) {
        return items;
    }

    let taskList: Awaited<ReturnType<typeof fetchImageTasks>>;
    try {
        taskList = await fetchImageTasks(taskIds);
    } catch {
        return items;
    }
    const taskMap = new Map(taskList.items.map((task) => [task.id, task]));
    let changed = false;
    const normalized = items.map((conversation) => {
        const turns = conversation.turns.map((turn) => {
            let turnChanged = false;
            const images = turn.images.map((image) => {
                if (image.status !== "loading" || !image.taskId) {
                    return image;
                }
                const task = taskMap.get(image.taskId);
                if (!task) {
                    return image;
                }
                const nextImage = taskDataToStoredImage(image, task);
                if (nextImage !== image) {
                    turnChanged = true;
                }
                return nextImage;
            });
            if (!turnChanged) {
                return turn;
            }
            changed = true;
            const derived = deriveTurnStatus({...turn, images});
            return {
                ...turn,
                ...derived,
                images,
            };
        });
        if (turns === conversation.turns || !turns.some((turn, index) => turn !== conversation.turns[index])) {
            return conversation;
        }
        return {
            ...conversation,
            turns,
            updatedAt: new Date().toISOString(),
        };
    });

    if (changed) {
        await saveImageConversations(normalized);
    }
    return normalized;
}

async function recoverConversationHistory(items: ImageConversation[]) {
    let changed = false;
    const normalized = items.map((conversation) => {
        const turns = conversation.turns.map((turn) => {
            if (turn.status !== "queued" && turn.status !== "generating") {
                return turn;
            }

            let turnChanged = false;
            const images = turn.images.map((image) => {
                if (image.status !== "loading" || image.taskId) {
                    return image;
                }
                turnChanged = true;
                return {
                    ...image,
                    status: "error" as const,
                    error: "页面刷新或任务中断，未找到可恢复的任务 ID",
                };
            });
            const derived = deriveTurnStatus({...turn, images});
            if (!turnChanged && derived.status === turn.status && derived.error === turn.error) {
                return turn;
            }
            changed = true;
            return {
                ...turn,
                ...derived,
                images,
            };
        });

        if (!turns.some((turn, index) => turn !== conversation.turns[index])) {
            return conversation;
        }

        return {
            ...conversation,
            turns,
            updatedAt: new Date().toISOString(),
        };
    });

    if (changed) {
        await saveImageConversations(normalized);
    }

    return syncConversationImageTasks(normalized);
}


function ImagePageContent({isAdmin, initialMaxImageCount}: { isAdmin: boolean; initialMaxImageCount: number }) {
    const didLoadQuotaRef = useRef(false);
    const conversationsRef = useRef<ImageConversation[]>([]);
    const resultsViewportRef = useRef<HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const [imagePrompt, setImagePrompt] = useState("");
    const [imageCount, setImageCount] = useState("1");
    const [imageSize, setImageSize] = useState("");
    const [imageResolution, setImageResolution] = useState<ImageResolution>("1k");
    const [isHistoryOpen, setIsHistoryOpen] = useState(false);
    const [referenceImageFiles, setReferenceImageFiles] = useState<File[]>([]);
    const [referenceImages, setReferenceImages] = useState<StoredReferenceImage[]>([]);
    const [conversations, setConversations] = useState<ImageConversation[]>([]);
    const [selectedConversationId, setSelectedConversationId] = useState<string | null>(null);
    const [isLoadingHistory, setIsLoadingHistory] = useState(true);
    const [availableQuota, setAvailableQuota] = useState("加载中...");
    const [maxImageCount, setMaxImageCount] = useState(initialMaxImageCount);
    const [isMaxImageCountLoaded, setIsMaxImageCountLoaded] = useState(false);
    const [lightboxImages, setLightboxImages] = useState<ImageLightboxItem[]>([]);
    const [lightboxOpen, setLightboxOpen] = useState(false);
    const [lightboxIndex, setLightboxIndex] = useState(0);
    const [deleteConfirm, setDeleteConfirm] = useState<
        | { type: "one"; id: string }
        | { type: "prompt"; conversationId: string; turnId: string }
        | { type: "results"; conversationId: string; turnId: string }
        | { type: "all" }
        | null
    >(null);
    const [continuousReference, setContinuousReference] = useState(false);

    const parsedCount = useMemo(() => Number(clampImageCount(imageCount, maxImageCount)), [imageCount, maxImageCount]);
    const effectiveSelectedConversationId = useMemo(() => {
        if (!selectedConversationId) {
            return null;
        }
        return conversations.some((item) => item.id === selectedConversationId)
            ? selectedConversationId
            : pickFallbackConversationId(conversations);
    }, [conversations, selectedConversationId]);
    const selectedConversation = useMemo(
        () => conversations.find((item) => item.id === effectiveSelectedConversationId) ?? null,
        [conversations, effectiveSelectedConversationId],
    );
    const activeTaskCount = useMemo(
        () =>
            conversations.reduce((sum, conversation) => {
                const stats = getImageConversationStats(conversation);
                return sum + stats.queued + stats.running;
            }, 0),
        [conversations],
    );
    const deleteConfirmTitle =
        deleteConfirm?.type === "all"
            ? "清空历史记录"
            : deleteConfirm?.type === "prompt"
                ? "删除提示词记录"
                : deleteConfirm?.type === "results"
                    ? "删除生成结果"
                    : deleteConfirm?.type === "one"
                        ? "删除对话"
                        : "";
    const deleteConfirmDescription =
        deleteConfirm?.type === "all"
            ? "确认删除全部图片历史记录吗？删除后无法恢复。"
            : deleteConfirm?.type === "prompt"
                ? "确认删除这条提示词记录吗？对应生成结果会保留。"
                : deleteConfirm?.type === "results"
                    ? "确认删除这条生成结果吗？对应提示词记录会保留。"
                    : deleteConfirm?.type === "one"
                        ? "确认删除这条图片对话吗？删除后无法恢复。"
                        : "";

    useEffect(() => {
        conversationsRef.current = conversations;
    }, [conversations]);

    useEffect(() => {
        let cancelled = false;

        const loadHistory = async () => {
            try {
                const storedSize = typeof window !== "undefined" ? window.localStorage.getItem(IMAGE_SIZE_STORAGE_KEY) : null;
                const storedResolution = typeof window !== "undefined" ? window.localStorage.getItem(IMAGE_RESOLUTION_STORAGE_KEY) : null;
                const storedCount = typeof window !== "undefined" ? window.localStorage.getItem(IMAGE_COUNT_STORAGE_KEY) : null;
                const storedContinuousReference = typeof window !== "undefined" ? window.localStorage.getItem(CONTINUOUS_REFERENCE_STORAGE_KEY) : null;
                setImageSize(storedSize || "");
                setImageResolution(normalizeImageResolution(storedResolution));
                setImageCount(storedCount || "1");
                setContinuousReference(storedContinuousReference === "true");

                const items = await listImageConversations();
                const normalizedItems = await recoverConversationHistory(items);
                if (cancelled) {
                    return;
                }

                conversationsRef.current = normalizedItems;
                setConversations(normalizedItems);
                const storedConversationId =
                    typeof window !== "undefined" ? window.localStorage.getItem(ACTIVE_CONVERSATION_STORAGE_KEY) : null;
                const nextSelectedConversationId =
                    (storedConversationId && normalizedItems.some((conversation) => conversation.id === storedConversationId)
                        ? storedConversationId
                        : null) ?? pickFallbackConversationId(normalizedItems);
                setSelectedConversationId(nextSelectedConversationId);
            } catch (error) {
                const message = error instanceof Error ? error.message : "读取会话记录失败";
                toast.error(message);
            } finally {
                if (!cancelled) {
                    setIsLoadingHistory(false);
                }
            }
        };

        void loadHistory();
        return () => {
            cancelled = true;
        };
    }, []);

    const loadQuota = useCallback(async () => {
        try {
            if (isAdmin) {
                setMaxImageCount(initialMaxImageCount);
                setIsMaxImageCountLoaded(true);
                setImageCount((prev) => (prev ? clampImageCount(prev, initialMaxImageCount) : prev));
                const data = await fetchAccounts();
                setAvailableQuota(formatAvailableQuota(data.items));
                return;
            }
            const identity = await fetchCurrentIdentity();
            const nextMaxImageCount = normalizeMaxImageCount(identity.image_task_max_count);
            setMaxImageCount(nextMaxImageCount);
            setIsMaxImageCountLoaded(true);
            setImageCount((prev) => (prev ? clampImageCount(prev, nextMaxImageCount) : prev));
            setAvailableQuota(formatUserQuota(identity.image_quota_available));
        } catch {
            setIsMaxImageCountLoaded(true);
            setAvailableQuota((prev) => (prev === "加载中..." ? "--" : prev));
        }
    }, [initialMaxImageCount, isAdmin]);

    useEffect(() => {
        if (didLoadQuotaRef.current) {
            return;
        }
        didLoadQuotaRef.current = true;

        const handleFocus = () => {
            void loadQuota();
        };

        void loadQuota();
        window.addEventListener("focus", handleFocus);
        return () => {
            window.removeEventListener("focus", handleFocus);
        };
    }, [isAdmin, loadQuota]);

    useEffect(() => {
        if (!selectedConversation) {
            return;
        }

        resultsViewportRef.current?.scrollTo({
            top: resultsViewportRef.current.scrollHeight,
            behavior: "smooth",
        });
    }, [selectedConversation?.updatedAt, selectedConversation?.turns.length, selectedConversation]);

    useEffect(() => {
        if (typeof window === "undefined") {
            return;
        }

        if (effectiveSelectedConversationId) {
            window.localStorage.setItem(ACTIVE_CONVERSATION_STORAGE_KEY, effectiveSelectedConversationId);
        } else {
            window.localStorage.removeItem(ACTIVE_CONVERSATION_STORAGE_KEY);
        }
    }, [effectiveSelectedConversationId]);

    useEffect(() => {
        if (typeof window === "undefined") {
            return;
        }

        if (imageSize) {
            window.localStorage.setItem(IMAGE_SIZE_STORAGE_KEY, imageSize);
            return;
        }
        window.localStorage.removeItem(IMAGE_SIZE_STORAGE_KEY);
    }, [imageSize]);

    useEffect(() => {
        if (typeof window !== "undefined") {
            window.localStorage.setItem(IMAGE_RESOLUTION_STORAGE_KEY, imageResolution);
        }
    }, [imageResolution]);

    useEffect(() => {
        if (typeof window !== "undefined" && isMaxImageCountLoaded && parsedCount > 0) {
            window.localStorage.setItem(IMAGE_COUNT_STORAGE_KEY, String(parsedCount));
        }
    }, [isMaxImageCountLoaded, parsedCount]);

    useEffect(() => {
        if (typeof window === "undefined") {
            return;
        }
        if (continuousReference) {
            window.localStorage.setItem(CONTINUOUS_REFERENCE_STORAGE_KEY, "true");
        } else {
            window.localStorage.removeItem(CONTINUOUS_REFERENCE_STORAGE_KEY);
        }
    }, [continuousReference]);

    const persistConversation = async (conversation: ImageConversation) => {
        const nextConversations = sortImageConversations([
            conversation,
            ...conversationsRef.current.filter((item) => item.id !== conversation.id),
        ]);
        conversationsRef.current = nextConversations;
        setConversations(nextConversations);
        await saveImageConversation(conversation);
    };

    const updateConversation = useCallback(
        async (
            conversationId: string,
            updater: (current: ImageConversation | null) => ImageConversation,
            options: { persist?: boolean } = {},
        ) => {
            const current = conversationsRef.current.find((item) => item.id === conversationId) ?? null;
            const nextConversation = updater(current);
            const nextConversations = sortImageConversations([
                nextConversation,
                ...conversationsRef.current.filter((item) => item.id !== conversationId),
            ]);
            conversationsRef.current = nextConversations;
            setConversations(nextConversations);
            if (options.persist !== false) {
                await saveImageConversation(nextConversation);
            }
        },
        [],
    );

    const clearComposerInputs = useCallback(() => {
        setImagePrompt("");
        setReferenceImageFiles([]);
        setReferenceImages([]);
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    }, []);

    const resetComposer = useCallback(() => {
        clearComposerInputs();
    }, [clearComposerInputs]);

    const handleCreateDraft = () => {
        setSelectedConversationId(null);
        resetComposer();
        textareaRef.current?.focus();
    };

    const handleDeleteConversation = async (id: string) => {
        const nextConversations = conversations.filter((item) => item.id !== id);
        conversationsRef.current = nextConversations;
        setConversations(nextConversations);
        if (selectedConversationId === id) {
            setSelectedConversationId(pickFallbackConversationId(nextConversations));
            resetComposer();
        }

        try {
            await deleteImageConversation(id);
        } catch (error) {
            const message = error instanceof Error ? error.message : "删除会话失败";
            toast.error(message);
            const items = await listImageConversations();
            conversationsRef.current = items;
            setConversations(items);
        }
    };

    const handleDeleteTurnPart = async (conversationId: string, turnId: string, part: "prompt" | "results") => {
        const conversation = conversationsRef.current.find((item) => item.id === conversationId);
        if (!conversation) {
            return;
        }

        const turns = conversation.turns
            .map((turn) => {
                if (turn.id !== turnId) {
                    return turn;
                }
                const nextTurn = {
                    ...turn,
                    prompt: part === "prompt" ? "" : turn.prompt,
                    promptDeleted: part === "prompt" ? true : turn.promptDeleted,
                    resultsDeleted: part === "results" ? true : turn.resultsDeleted,
                    status: part === "results" && turn.status === "generating" ? "error" as const : turn.status,
                    images:
                        part === "results"
                            ? turn.images.map((image) => ({
                                id: image.id,
                                status: "error" as const,
                                error: "生成结果已删除"
                            }))
                            : turn.images,
                };
                return nextTurn.promptDeleted && nextTurn.resultsDeleted ? null : nextTurn;
            })
            .filter((turn): turn is ImageTurn => Boolean(turn));

        if (turns.length === 0) {
            await handleDeleteConversation(conversationId);
            return;
        }

        const nextConversation = {
            ...conversation,
            updatedAt: new Date().toISOString(),
            turns,
        };
        await persistConversation(nextConversation);
    };

    const handleClearHistory = async () => {
        try {
            await clearImageConversations();
            conversationsRef.current = [];
            setConversations([]);
            setSelectedConversationId(null);
            resetComposer();
            toast.success("已清空历史记录");
        } catch (error) {
            const message = error instanceof Error ? error.message : "清空历史记录失败";
            toast.error(message);
        }
    };

    const handleRenameConversation = async (id: string, title: string) => {
        const nextConversations = conversations.map((item) =>
            item.id === id ? {...item, title, updatedAt: new Date().toISOString()} : item,
        );
        conversationsRef.current = sortImageConversations(nextConversations);
        setConversations(conversationsRef.current);
        try {
            await renameImageConversation(id, title);
        } catch (error) {
            const message = error instanceof Error ? error.message : "重命名失败";
            toast.error(message);
        }
    };

    const openDeleteConversationConfirm = (id: string) => {
        setIsHistoryOpen(false);
        setDeleteConfirm({type: "one", id});
    };

    const openDeletePromptConfirm = (conversationId: string, turnId: string) => {
        setDeleteConfirm({type: "prompt", conversationId, turnId});
    };

    const openDeleteResultsConfirm = (conversationId: string, turnId: string) => {
        setDeleteConfirm({type: "results", conversationId, turnId});
    };

    const openClearHistoryConfirm = () => {
        setIsHistoryOpen(false);
        setDeleteConfirm({type: "all"});
    };

    const handleConfirmDelete = async () => {
        const target = deleteConfirm;
        setDeleteConfirm(null);
        if (!target) {
            return;
        }
        if (target.type === "all") {
            await handleClearHistory();
            return;
        }
        if (target.type === "prompt" || target.type === "results") {
            await handleDeleteTurnPart(target.conversationId, target.turnId, target.type);
            return;
        }
        await handleDeleteConversation(target.id);
    };

    const appendReferenceImages = useCallback(async (files: File[]) => {
        if (files.length === 0) {
            return;
        }

        try {
            const previews = await Promise.all(
                files.map(async (file) => ({
                    name: file.name,
                    type: file.type || "image/png",
                    dataUrl: await readFileAsDataUrl(file),
                })),
            );

            setReferenceImageFiles((prev) => [...prev, ...files]);
            setReferenceImages((prev) => [...prev, ...previews]);
            if (fileInputRef.current) {
                fileInputRef.current.value = "";
            }
        } catch (error) {
            const message = error instanceof Error ? error.message : "读取参考图失败";
            toast.error(message);
        }
    }, []);

    const handleReferenceImageChange = useCallback(
        async (files: File[]) => {
            if (files.length === 0) {
                return;
            }

            await appendReferenceImages(files);
        },
        [appendReferenceImages],
    );

    const handleRemoveReferenceImage = useCallback((index: number) => {
        setReferenceImageFiles((prev) => {
            const next = prev.filter((_, currentIndex) => currentIndex !== index);
            if (next.length === 0 && fileInputRef.current) {
                fileInputRef.current.value = "";
            }
            return next;
        });
        setReferenceImages((prev) => prev.filter((_, currentIndex) => currentIndex !== index));
    }, []);

    const handleContinueEdit = useCallback(
        async (conversationId: string, image: StoredImage | StoredReferenceImage) => {
            try {
                const nextReference =
                    "dataUrl" in image
                        ? {
                            referenceImage: image,
                            file: dataUrlToFile(image.dataUrl, image.name, image.type),
                        }
                        : await buildReferenceImageFromStoredImage(image, `conversation-${conversationId}-${Date.now()}.png`);
                if (!nextReference) {
                    return;
                }

                setSelectedConversationId(conversationId);

                setReferenceImages((prev) => [...prev, nextReference.referenceImage]);
                setReferenceImageFiles((prev) => [...prev, nextReference.file]);
                setImagePrompt("");
                textareaRef.current?.focus();
                toast.success("已加入当前参考图，继续输入描述即可编辑");
            } catch (error) {
                const message = error instanceof Error ? error.message : "读取结果图失败";
                toast.error(message);
            }
        },
        [],
    );

    const handleReuseTurnConfig = useCallback(async (conversationId: string, turnId: string) => {
        const conversation = conversationsRef.current.find((item) => item.id === conversationId);
        const turn = conversation?.turns.find((item) => item.id === turnId);
        if (!conversation || !turn || !turn.prompt.trim()) {
            return;
        }

        setSelectedConversationId(conversationId);
        setImagePrompt(turn.prompt);
        setImageCount(clampImageCount(String(turn.count || turn.images.length || 1), maxImageCount));
        setImageSize(turn.size);
        setImageResolution(normalizeImageResolution(turn.resolution));
        setReferenceImages(turn.referenceImages);
        setReferenceImageFiles(
            turn.referenceImages.map((image) => dataUrlToFile(image.dataUrl, image.name, image.type)),
        );
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
        textareaRef.current?.focus();
        toast.success("已复用这条提示词配置");
    }, [maxImageCount]);

    const openLightbox = useCallback((images: ImageLightboxItem[], index: number) => {
        if (images.length === 0) {
            return;
        }

        setLightboxImages(images);
        setLightboxIndex(Math.max(0, Math.min(index, images.length - 1)));
        setLightboxOpen(true);
    }, []);

    const createLoadingImages = (turnId: string, count: number, resolution: ImageResolution) =>
        Array.from({length: count}, (_, index) => {
            const imageId = `${turnId}-${index}`;
            return {
                id: imageId,
                taskId: imageId,
                resolution,
                status: "loading" as const,
            };
        });

    /* eslint-disable react-hooks/preserve-manual-memoization */
    const runConversationQueue = useCallback(
        async (conversationId: string) => {
            if (activeConversationQueueIds.has(conversationId)) {
                return;
            }

            const snapshot = conversationsRef.current.find((conversation) => conversation.id === conversationId);
            const activeTurn = snapshot?.turns.find(
                (turn) =>
                    (turn.status === "queued" || turn.status === "generating") &&
                    turn.images.some((image) => image.status === "loading"),
            );
            if (!snapshot || !activeTurn) {
                return;
            }

            activeConversationQueueIds.add(conversationId);
            const applyResolutionFallbacks = async (
                fallbacks: Array<{
                    taskId: string;
                    currentResolution: ImageResolution;
                    nextResolution: ImageResolution;
                }>,
            ) => {
                if (fallbacks.length === 0) {
                    return;
                }
                const fallbackMap = new Map(fallbacks.map((item) => [item.taskId, item]));
                await updateConversation(conversationId, (current) => {
                    const conversation = current ?? snapshot;
                    const turns = conversation.turns.map((turn) => {
                        if (turn.id !== activeTurn.id) {
                            return turn;
                        }
                        const images = turn.images.map((image) => {
                            const taskId = image.taskId || image.id;
                            const fallback = fallbackMap.get(taskId);
                            if (!fallback || image.status !== "loading") {
                                return image;
                            }
                            return {
                                ...image,
                                taskId: fallbackTaskId(image.id, fallback.nextResolution),
                                resolution: fallback.nextResolution,
                                status: "loading" as const,
                                error: undefined,
                            };
                        });
                        const derived = deriveTurnStatus({...turn, status: "generating", images});
                        return {
                            ...turn,
                            ...derived,
                            images,
                        };
                    });
                    return {
                        ...conversation,
                        updatedAt: new Date().toISOString(),
                        turns,
                    };
                });
                const messages = fallbacks.map(
                    (item) =>
                        `当前清晰度偏好不可用，已从 ${imageResolutionLabel(item.currentResolution)} 降级到 ${imageResolutionLabel(item.nextResolution)} 重新生成`,
                );
                Array.from(new Set(messages)).forEach((message) => toast.warning(message));
            };
            const applyTasks = async (tasks: ImageTask[]) => {
                const taskMap = new Map(tasks.map((task) => [task.id, task]));
                const resolutionFallbacks: Array<{
                    taskId: string;
                    currentResolution: ImageResolution;
                    nextResolution: ImageResolution;
                }> = [];
                await updateConversation(conversationId, (current) => {
                    const conversation = current ?? snapshot;
                    const turns = conversation.turns.map((turn) => {
                        if (turn.id !== activeTurn.id) {
                            return turn;
                        }
                        const images = turn.images.map((image) => {
                            const taskId = image.taskId || image.id;
                            const task = taskMap.get(taskId);
                            if (!task) {
                                return image;
                            }
                            const currentResolution = normalizeImageResolution(image.resolution || turn.resolution);
                            const nextResolution =
                                task.status === "error" && isResolutionFallbackError(task.error)
                                    ? nextLowerImageResolution(currentResolution)
                                    : null;
                            if (nextResolution) {
                                resolutionFallbacks.push({taskId, currentResolution, nextResolution});
                                return image;
                            }
                            return taskDataToStoredImage({...image, taskId, resolution: currentResolution}, task);
                        });
                        const derived = deriveTurnStatus({...turn, status: "generating", images});
                        return {
                            ...turn,
                            ...derived,
                            images,
                        };
                    });
                    return {
                        ...conversation,
                        updatedAt: new Date().toISOString(),
                        turns,
                    };
                });
                await applyResolutionFallbacks(resolutionFallbacks);
            };

            try {
                await updateConversation(conversationId, (current) => {
                    const conversation = current ?? snapshot;
                    return {
                        ...conversation,
                        updatedAt: new Date().toISOString(),
                        turns: conversation.turns.map((turn) =>
                            turn.id === activeTurn.id
                                ? {
                                    ...turn,
                                    status: "generating",
                                    error: undefined,
                                    images: turn.images.map((image) =>
                                        image.status === "loading" ? {...image, taskId: image.taskId || image.id} : image,
                                    ),
                                }
                                : turn,
                        ),
                    };
                });

                const referenceFiles = activeTurn.referenceImages.map((image, index) =>
                    dataUrlToFile(image.dataUrl, image.name || `${activeTurn.id}-${index + 1}.png`, image.type),
                );
                const taskPrompt = buildContextualImagePrompt(snapshot, activeTurn);
                ensureEditableReferences(activeTurn, referenceFiles);

                const markSubmitFailures = async (failures: Array<{ taskId: string; message: string }>) => {
                    if (failures.length === 0) {
                        return;
                    }
                    const errorMap = new Map(failures.map((item) => [item.taskId, item.message]));
                    await updateConversation(conversationId, (current) => {
                        const conversation = current ?? snapshot;
                        const turns = conversation.turns.map((turn) => {
                            if (turn.id !== activeTurn.id) {
                                return turn;
                            }
                            const images = turn.images.map((image) => {
                                const taskId = image.taskId || image.id;
                                const message = errorMap.get(taskId);
                                if (!message || image.status !== "loading") {
                                    return image;
                                }
                                return {
                                    ...image,
                                    taskId,
                                    status: "error" as const,
                                    error: message,
                                };
                            });
                            const derived = deriveTurnStatus({...turn, images});
                            return {
                                ...turn,
                                ...derived,
                                images,
                            };
                        });
                        return {
                            ...conversation,
                            updatedAt: new Date().toISOString(),
                            turns,
                        };
                    });
                };

                const submitImageTasks = async (images: StoredImage[]) => {
                    const submittedTasks: ImageTask[] = [];
                    const failures: Array<{ taskId: string; message: string }> = [];
                    const resolutionFallbacks: Array<{
                        taskId: string;
                        currentResolution: ImageResolution;
                        nextResolution: ImageResolution;
                    }> = [];
                    await forEachWithConcurrency(
                        images,
                        SUBMIT_IMAGE_TASK_CONCURRENCY,
                        async (image) => {
                            const taskId = image.taskId || image.id;
                            const resolution = normalizeImageResolution(image.resolution || activeTurn.resolution);
                            try {
                                const task = activeTurn.mode === "edit"
                                    ? await createImageEditTask(taskId, referenceFiles, taskPrompt, activeTurn.model, activeTurn.size, resolution)
                                    : await createImageGenerationTask(taskId, taskPrompt, activeTurn.model, activeTurn.size, resolution);
                                submittedTasks.push(task);
                            } catch (error) {
                                const message = error instanceof Error ? error.message : "创建图片任务失败";
                                const nextResolution = isResolutionFallbackError(message) ? nextLowerImageResolution(resolution) : null;
                                if (nextResolution) {
                                    resolutionFallbacks.push({taskId, currentResolution: resolution, nextResolution});
                                    return;
                                }
                                failures.push({
                                    taskId,
                                    message,
                                });
                            }
                        },
                    );
                    if (submittedTasks.length > 0) {
                        await applyTasks(submittedTasks);
                    }
                    await applyResolutionFallbacks(resolutionFallbacks);
                    await markSubmitFailures(failures);
                    if (failures.length > 0) {
                        const uniqueMessages = Array.from(new Set(failures.map((item) => item.message)));
                        toast.error(uniqueMessages.length === 1 ? uniqueMessages[0] : `${failures.length} 张图片任务创建失败`);
                    }
                };

                const pendingImages = activeTurn.images.filter((image) => image.status === "loading");
                await submitImageTasks(pendingImages);

                while (true) {
                    const latestConversation = conversationsRef.current.find((conversation) => conversation.id === conversationId);
                    const latestTurn = latestConversation?.turns.find((turn) => turn.id === activeTurn.id);
                    const loadingTaskIds =
                        latestTurn?.images.flatMap((image) =>
                            image.status === "loading" && image.taskId ? [image.taskId] : [],
                        ) || [];
                    if (loadingTaskIds.length === 0) {
                        break;
                    }

                    await sleep(2000);
                    const taskList = await fetchImageTasks(loadingTaskIds);
                    if (taskList.items.length > 0) {
                        await applyTasks(taskList.items);
                    }
                    if (taskList.missing_ids.length > 0 && latestTurn) {
                        const missingImages = latestTurn.images.filter(
                            (image) => image.status === "loading" && image.taskId && taskList.missing_ids.includes(image.taskId),
                        );
                        await submitImageTasks(missingImages);
                    }
                }

                await loadQuota();
            } catch (error) {
                const message = error instanceof Error ? error.message : "生成图片失败";
                await updateConversation(conversationId, (current) => {
                    const conversation = current ?? snapshot;
                    return {
                        ...conversation,
                        updatedAt: new Date().toISOString(),
                        turns: conversation.turns.map((turn) =>
                            turn.id === activeTurn.id
                                ? {
                                    ...turn,
                                    status: "error",
                                    error: message,
                                    images: turn.images.map((image) =>
                                        image.status === "loading" ? {...image, status: "error", error: message} : image,
                                    ),
                                }
                                : turn,
                        ),
                    };
                });
                toast.error(message);
            } finally {
                activeConversationQueueIds.delete(conversationId);
                for (const conversation of conversationsRef.current) {
                    if (
                        !activeConversationQueueIds.has(conversation.id) &&
                        conversation.turns.some(
                            (turn) =>
                                (turn.status === "queued" || turn.status === "generating") &&
                                turn.images.some((image) => image.status === "loading"),
                        )
                    ) {
                        void runConversationQueue(conversation.id);
                    }
                }
            }
        },
        [loadQuota, updateConversation],
    );
    /* eslint-enable react-hooks/preserve-manual-memoization */

    const handleRegenerateTurn = useCallback(
        async (conversationId: string, turnId: string) => {
            const conversation = conversationsRef.current.find((item) => item.id === conversationId);
            const sourceTurn = conversation?.turns.find((turn) => turn.id === turnId);
            if (!conversation || !sourceTurn || !sourceTurn.prompt.trim()) {
                return;
            }

            const now = new Date().toISOString();
            const nextTurnId = createId();
            const count = Number(clampImageCount(String(sourceTurn.count || sourceTurn.images.length || 1), maxImageCount));
            const automaticReferences = sourceTurn.referenceImages.length > 0 || !continuousReference
                ? {referenceImages: [] as StoredReferenceImage[], referenceFiles: [] as File[]}
                : await buildAutomaticReferenceImages(conversation, sourceTurn.id);
            const nextTurn: ImageTurn = {
                id: nextTurnId,
                prompt: sourceTurn.prompt,
                model: sourceTurn.model,
                mode: sourceTurn.referenceImages.length > 0 ? "edit" : automaticReferences.referenceImages.length > 0 ? "edit" : sourceTurn.mode,
                referenceImages:
                    sourceTurn.referenceImages.length > 0 ? sourceTurn.referenceImages : automaticReferences.referenceImages,
                count,
                size: sourceTurn.size,
                resolution: normalizeImageResolution(sourceTurn.resolution),
                images: createLoadingImages(nextTurnId, count, normalizeImageResolution(sourceTurn.resolution)),
                createdAt: now,
                status: "queued",
            };
            const nextConversation = {
                ...conversation,
                updatedAt: now,
                turns: [...conversation.turns, nextTurn],
            };

            setSelectedConversationId(conversationId);
            await persistConversation(nextConversation);
            void runConversationQueue(conversationId);
            toast.success("已加入重新生成队列");
        },
        [maxImageCount, runConversationQueue, continuousReference],
    );

    const handleRetryImage = useCallback(
        async (conversationId: string, turnId: string, imageId: string) => {
            const conversation = conversationsRef.current.find((item) => item.id === conversationId);
            if (!conversation) {
                return;
            }

            const now = new Date().toISOString();
            const retryImageId = `${turnId}-${createId()}`;
            const nextConversation = {
                ...conversation,
                updatedAt: now,
                turns: conversation.turns.map((turn) => {
                    if (turn.id !== turnId) {
                        return turn;
                    }
                    if (!turn.prompt.trim()) {
                        return turn;
                    }

                    const images = turn.images.map((image) =>
                        image.id === imageId
                            ? {
                                id: retryImageId,
                                taskId: retryImageId,
                                resolution: normalizeImageResolution(turn.resolution),
                                status: "loading" as const,
                            }
                            : image,
                    );
                    const derived = deriveTurnStatus({...turn, status: "queued", images});
                    return {
                        ...turn,
                        ...derived,
                        images,
                    };
                }),
            };

            setSelectedConversationId(conversationId);
            await persistConversation(nextConversation);
            void runConversationQueue(conversationId);
        },
        [runConversationQueue],
    );

    useEffect(() => {
        for (const conversation of conversations) {
            if (
                !activeConversationQueueIds.has(conversation.id) &&
                conversation.turns.some(
                    (turn) =>
                        !turn.resultsDeleted &&
                        (turn.status === "queued" || turn.status === "generating") &&
                        turn.images.some((image) => image.status === "loading"),
                )
            ) {
                void runConversationQueue(conversation.id);
            }
        }
    }, [conversations, runConversationQueue]);

    const handleSubmit = async () => {
        const prompt = imagePrompt.trim();
        if (!prompt) {
            toast.error("请输入提示词");
            return;
        }

        const targetConversation = effectiveSelectedConversationId
            ? conversationsRef.current.find((conversation) => conversation.id === effectiveSelectedConversationId) ?? null
            : null;
        const automaticReferences =
            referenceImageFiles.length > 0 || !continuousReference
                ? {referenceImages: [] as StoredReferenceImage[], referenceFiles: [] as File[]}
                : await buildAutomaticReferenceImages(targetConversation);
        const nextReferenceImages = referenceImages.length > 0 ? referenceImages : automaticReferences.referenceImages;
        const nextReferenceFiles = referenceImageFiles.length > 0 ? referenceImageFiles : automaticReferences.referenceFiles;
        const nextImageMode: ImageConversationMode = nextReferenceFiles.length > 0 ? "edit" : "generate";
        const now = new Date().toISOString();
        const conversationId = targetConversation?.id ?? createId();
        const turnId = createId();
        const draftTurn: ImageTurn = {
            id: turnId,
            prompt,
            model: "gpt-image-2",
            mode: nextImageMode,
            referenceImages: nextImageMode === "edit" ? nextReferenceImages : [],
            count: parsedCount,
            size: imageSize,
            resolution: imageResolution,
            images: createLoadingImages(turnId, parsedCount, imageResolution),
            createdAt: now,
            status: "queued",
        };

        const baseConversation: ImageConversation = targetConversation
            ? {
                ...targetConversation,
                updatedAt: now,
                turns: [...targetConversation.turns, draftTurn],
            }
            : {
                id: conversationId,
                title: buildConversationTitle(prompt),
                createdAt: now,
                updatedAt: now,
                turns: [draftTurn],
            };

        setSelectedConversationId(conversationId);
        clearComposerInputs();

        await persistConversation(baseConversation);
        void runConversationQueue(conversationId);

        const targetStats = getImageConversationStats(baseConversation);
        if (targetStats.running > 0 || targetStats.queued > 1) {
            toast.success("已加入当前对话队列");
        } else if (!targetConversation) {
            toast.success("已创建新对话并开始处理");
        } else {
            toast.success("已发送到当前对话");
        }
    };

    return (
        <>
            <section
                className="mx-auto grid h-[calc(100dvh-6.5rem)] min-h-0 w-full max-w-[1380px] grid-cols-1 gap-2 overflow-hidden px-0 pb-[calc(env(safe-area-inset-bottom)+0.5rem)] sm:h-[calc(100dvh-5.25rem)] sm:gap-3 sm:px-3 sm:pb-6 lg:grid-cols-[240px_minmax(0,1fr)]">
                <div className="hidden h-full min-h-0 border-r border-stone-200/70 pr-3 lg:block">
                    <ImageSidebar
                        conversations={conversations}
                        isLoadingHistory={isLoadingHistory}
                        selectedConversationId={effectiveSelectedConversationId}
                        onCreateDraft={handleCreateDraft}
                        onClearHistory={openClearHistoryConfirm}
                        onSelectConversation={setSelectedConversationId}
                        onDeleteConversation={openDeleteConversationConfirm}
                        onRenameConversation={handleRenameConversation}
                        formatConversationTime={formatConversationTime}
                    />
                </div>

                <Dialog open={isHistoryOpen} onOpenChange={setIsHistoryOpen}>
                    <DialogContent
                        className="flex h-[min(82dvh,760px)] w-[92vw] max-w-[460px] flex-col overflow-hidden rounded-[32px] border-white/80 bg-white p-0 shadow-[0_32px_110px_-38px_rgba(15,23,42,0.45)] sm:rounded-[36px]">
                        <DialogHeader className="px-6 pt-7 pb-4 sm:px-8">
                            <DialogTitle className="flex items-center gap-2 text-xl font-bold tracking-tight">
                                <History className="size-5"/>
                                历史记录
                            </DialogTitle>
                        </DialogHeader>
                        <div className="min-h-0 flex-1 overflow-y-auto px-5 pb-8 sm:px-8">
                            <ImageSidebar
                                conversations={conversations}
                                isLoadingHistory={isLoadingHistory}
                                selectedConversationId={effectiveSelectedConversationId}
                                onCreateDraft={() => {
                                    handleCreateDraft();
                                    setIsHistoryOpen(false);
                                }}
                                onClearHistory={openClearHistoryConfirm}
                                onSelectConversation={(id) => {
                                    setSelectedConversationId(id);
                                    setIsHistoryOpen(false);
                                }}
                                onDeleteConversation={openDeleteConversationConfirm}
                                onRenameConversation={handleRenameConversation}
                                formatConversationTime={formatConversationTime}
                                hideActionButtons
                            />
                        </div>
                    </DialogContent>
                </Dialog>

                <div className="flex min-h-0 flex-col gap-2 sm:gap-4">
                    <div className="flex items-center justify-between gap-2 px-1 lg:hidden">
                        <Button
                            variant="outline"
                            className="h-10 flex-1 rounded-2xl border-stone-200 bg-white/90 text-stone-700 shadow-sm"
                            onClick={() => setIsHistoryOpen(true)}
                        >
                            <History className="mr-2 size-4"/>
                            历史记录 ({conversations.length})
                        </Button>
                        <Button
                            className="h-10 rounded-2xl bg-stone-950 text-white shadow-sm"
                            onClick={handleCreateDraft}
                        >
                            <Plus className="size-4"/>
                            新建
                        </Button>
                        <Button
                            variant="outline"
                            className="h-10 rounded-2xl border-stone-200 bg-white/85 px-3 text-stone-600 shadow-sm"
                            onClick={openClearHistoryConfirm}
                            disabled={conversations.length === 0}
                        >
                            <Trash2 className="size-4"/>
                        </Button>
                    </div>

                    <div
                        ref={resultsViewportRef}
                        className="hide-scrollbar min-h-0 flex-1 overscroll-contain overflow-y-auto px-1 py-2 sm:px-4 sm:py-4"
                    >
                        <ImageResults
                            selectedConversation={selectedConversation}
                            onOpenLightbox={openLightbox}
                            onContinueEdit={handleContinueEdit}
                            onDeletePrompt={openDeletePromptConfirm}
                            onDeleteResults={openDeleteResultsConfirm}
                            onReuseTurnConfig={handleReuseTurnConfig}
                            onRegenerateTurn={handleRegenerateTurn}
                            onRetryImage={handleRetryImage}
                            formatConversationTime={formatConversationTime}
                        />
                    </div>

                    <ImageComposer
                        prompt={imagePrompt}
                        imageCount={imageCount}
                        imageSize={imageSize}
                        imageResolution={imageResolution}
                        availableQuota={availableQuota}
                        activeTaskCount={activeTaskCount}
                        maxImageCount={maxImageCount}
                        referenceImages={referenceImages}
                        textareaRef={textareaRef}
                        fileInputRef={fileInputRef}
                        onPromptChange={setImagePrompt}
                        onImageCountChange={(value) => setImageCount(value ? clampImageCount(value, maxImageCount) : "")}
                        onImageSizeChange={setImageSize}
                        onImageResolutionChange={setImageResolution}
                        onSubmit={handleSubmit}
                        onPickReferenceImage={() => fileInputRef.current?.click()}
                        onReferenceImageChange={handleReferenceImageChange}
                        onRemoveReferenceImage={handleRemoveReferenceImage}
                        continuousReference={continuousReference}
                        onContinuousReferenceChange={setContinuousReference}
                    />
                </div>
            </section>

            <ImageLightbox
                images={lightboxImages}
                currentIndex={lightboxIndex}
                open={lightboxOpen}
                onOpenChange={setLightboxOpen}
                onIndexChange={setLightboxIndex}
            />

            {deleteConfirm ? (
                <Dialog open onOpenChange={(open) => (!open ? setDeleteConfirm(null) : null)}>
                    <DialogContent showCloseButton={false} className="rounded-2xl p-6">
                        <DialogHeader className="gap-2">
                            <DialogTitle>{deleteConfirmTitle}</DialogTitle>
                            <DialogDescription className="text-sm leading-6">
                                {deleteConfirmDescription}
                            </DialogDescription>
                        </DialogHeader>
                        <DialogFooter>
                            <Button variant="outline" onClick={() => setDeleteConfirm(null)}>
                                取消
                            </Button>
                            <Button className="bg-rose-600 text-white hover:bg-rose-700"
                                    onClick={() => void handleConfirmDelete()}>
                                确认删除
                            </Button>
                        </DialogFooter>
                    </DialogContent>
                </Dialog>
            ) : null}
        </>
    );
}

export default function ImagePage() {
    const {isCheckingAuth, session} = useAuthGuard();

    if (isCheckingAuth || !session) {
        return (
            <div className="flex min-h-[40vh] items-center justify-center">
                <LoaderCircle className="size-5 animate-spin text-stone-400"/>
            </div>
        );
    }

    return (
        <ImagePageContent
            isAdmin={session.role === "admin"}
            initialMaxImageCount={normalizeMaxImageCount(session.imageTaskMaxCount)}
        />
    );
}

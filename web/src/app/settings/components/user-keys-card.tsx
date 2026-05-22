"use client";

import {useEffect, useRef, useState} from "react";
import {Ban, CheckCircle2, Coins, Copy, KeyRound, LoaderCircle, Pencil, Plus, Trash2} from "lucide-react";
import {toast} from "sonner";

import {Badge} from "@/components/ui/badge";
import {Button} from "@/components/ui/button";
import {Card, CardContent} from "@/components/ui/card";
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from "@/components/ui/dialog";
import {Input} from "@/components/ui/input";
import {Textarea} from "@/components/ui/textarea";
import {
    createUserKey,
    deleteUserKey,
    fetchUserKeys,
    rechargeUserKeyQuota,
    updateUserKey,
    type UserKey
} from "@/lib/api";

function formatDateTime(value?: string | null) {
    if (!value) {
        return "—";
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return value;
    }
    return new Intl.DateTimeFormat("zh-CN", {
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
    }).format(date);
}

function formatQuota(value?: number | null) {
    const amount = Number(value ?? 0);
    if (!Number.isFinite(amount)) {
        return "0";
    }
    return new Intl.NumberFormat("zh-CN", {maximumFractionDigits: 6}).format(amount);
}

function normalizeAllowedIps(value: string) {
    return Array.from(new Set(value.split(/[\s,]+/).map((item) => item.trim()).filter(Boolean)));
}

function buildUserKeyDemoCurl(baseUrl: string, apiKey: string) {
    const normalizedBaseUrl = baseUrl.trim().replace(/\/+$/, "") || "https://your-domain.example/v1";
    const endpoint = `${normalizedBaseUrl.endsWith("/v1") ? normalizedBaseUrl : `${normalizedBaseUrl}/v1`}/images/generations`;

    return `curl --request POST "${endpoint}" \\
  --header "Authorization: Bearer ${apiKey}" \\
  --header "Content-Type: application/json" \\
  --data '{
    "model": "gpt-image-2",
    "prompt": "一只橘猫坐在电脑前测试接口，可爱插画风格",
    "n": 1,
    "response_format": "b64_json"
  }'`;
}

export function UserKeysCard() {
    const didLoadRef = useRef(false);
    const [items, setItems] = useState<UserKey[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [isDialogOpen, setIsDialogOpen] = useState(false);
    const [name, setName] = useState("");
    const [imageQuota, setImageQuota] = useState("0");
    const [allowedIps, setAllowedIps] = useState("");
    const [isCreating, setIsCreating] = useState(false);
    const [pendingIds, setPendingIds] = useState<Set<string>>(() => new Set());
    const [revealedKey, setRevealedKey] = useState("");
    const [deletingItem, setDeletingItem] = useState<UserKey | null>(null);
    const [editingItem, setEditingItem] = useState<UserKey | null>(null);
    const [rechargingItem, setRechargingItem] = useState<UserKey | null>(null);
    const [editName, setEditName] = useState("");
    const [editKey, setEditKey] = useState("");
    const [editAllowedIps, setEditAllowedIps] = useState("");
    const [rechargeAmount, setRechargeAmount] = useState("");
    const [isRecharging, setIsRecharging] = useState(false);
    const [demoBaseUrl] = useState(() =>
        typeof window !== "undefined" ? `${window.location.origin}/v1` : "https://your-domain.example/v1",
    );

    const load = async () => {
        setIsLoading(true);
        try {
            const data = await fetchUserKeys();
            setItems(data.items);
        } catch (error) {
            toast.error(error instanceof Error ? error.message : "加载用户密钥失败");
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        if (didLoadRef.current) {
            return;
        }
        didLoadRef.current = true;
        void load();
    }, []);

    const handleCreate = async () => {
        const quota = Math.max(0, Number(imageQuota) || 0);
        setIsCreating(true);
        try {
            const data = await createUserKey(name.trim(), quota, normalizeAllowedIps(allowedIps));
            setItems(data.items);
            setRevealedKey(data.key);
            setName("");
            setImageQuota("0");
            setAllowedIps("");
            setIsDialogOpen(false);
            toast.success("用户密钥已创建");
        } catch (error) {
            toast.error(error instanceof Error ? error.message : "创建用户密钥失败");
        } finally {
            setIsCreating(false);
        }
    };

    const setItemPending = (id: string, isPending: boolean) => {
        setPendingIds((current) => {
            const next = new Set(current);
            if (isPending) {
                next.add(id);
            } else {
                next.delete(id);
            }
            return next;
        });
    };

    const handleToggle = async (item: UserKey) => {
        setItemPending(item.id, true);
        try {
            const data = await updateUserKey(item.id, {enabled: !item.enabled});
            setItems(data.items);
            toast.success(item.enabled ? "用户密钥已禁用" : "用户密钥已启用");
        } catch (error) {
            toast.error(error instanceof Error ? error.message : "更新用户密钥失败");
        } finally {
            setItemPending(item.id, false);
        }
    };

    const handleDelete = async () => {
        if (!deletingItem) {
            return;
        }
        const item = deletingItem;
        setItemPending(item.id, true);
        try {
            const data = await deleteUserKey(item.id);
            setItems(data.items);
            setDeletingItem(null);
            toast.success("用户密钥已删除");
        } catch (error) {
            toast.error(error instanceof Error ? error.message : "删除用户密钥失败");
        } finally {
            setItemPending(item.id, false);
        }
    };

    const openEditDialog = (item: UserKey) => {
        setEditingItem(item);
        setEditName(item.name);
        setEditKey("");
        setEditAllowedIps((item.allowed_ips || []).join("\n"));
    };

    const openRechargeDialog = (item: UserKey) => {
        setRechargingItem(item);
        setRechargeAmount("");
    };

    const handleRecharge = async () => {
        if (!rechargingItem) {
            return;
        }
        const amount = Number(rechargeAmount);
        if (!Number.isFinite(amount) || amount <= 0) {
            toast.error("请输入大于 0 的充值额度");
            return;
        }
        const item = rechargingItem;
        setIsRecharging(true);
        setItemPending(item.id, true);
        try {
            const data = await rechargeUserKeyQuota(item.id, amount);
            setItems(data.items);
            setRechargingItem(null);
            setRechargeAmount("");
            toast.success("用户密钥额度已充值");
        } catch (error) {
            toast.error(error instanceof Error ? error.message : "充值用户密钥额度失败");
        } finally {
            setIsRecharging(false);
            setItemPending(item.id, false);
        }
    };

    const handleEdit = async () => {
        if (!editingItem) {
            return;
        }
        const item = editingItem;
        const trimmedName = editName.trim();
        const trimmedKey = editKey.trim();
        const nextAllowedIps = normalizeAllowedIps(editAllowedIps);
        const currentAllowedIps = item.allowed_ips || [];
        const allowedIpsChanged = nextAllowedIps.join("\n") !== currentAllowedIps.join("\n");
        if (trimmedName === item.name && !trimmedKey && !allowedIpsChanged) {
            setEditingItem(null);
            return;
        }
        setItemPending(item.id, true);
        try {
            const data = await updateUserKey(item.id, {
                ...(trimmedName !== item.name ? {name: trimmedName} : {}),
                ...(trimmedKey ? {key: trimmedKey} : {}),
                ...(allowedIpsChanged ? {allowed_ips: nextAllowedIps} : {}),
            });
            setItems(data.items);
            setEditingItem(null);
            setEditKey("");
            toast.success(trimmedKey ? "用户密钥已更新" : "用户名称已更新");
        } catch (error) {
            toast.error(error instanceof Error ? error.message : "更新用户密钥失败");
        } finally {
            setItemPending(item.id, false);
        }
    };

    const handleCopy = async (value: string) => {
        try {
            await navigator.clipboard.writeText(value);
            toast.success("已复制到剪贴板");
        } catch {
            toast.error("复制失败，请手动复制");
        }
    };

    const revealedKeyDemoCurl = revealedKey ? buildUserKeyDemoCurl(demoBaseUrl, revealedKey) : "";

    return (
        <>
            <Card className="rounded-2xl border-white/80 bg-white/90 shadow-sm">
                <CardContent className="space-y-6 p-6">
                    <div className="flex items-start justify-between gap-4">
                        <div className="flex items-center gap-3">
                            <div className="flex size-10 items-center justify-center rounded-xl bg-stone-100">
                                <KeyRound className="size-5 text-stone-600"/>
                            </div>
                            <div>
                                <h2 className="text-lg font-semibold tracking-tight">用户密钥管理</h2>
                                <p className="text-sm text-stone-500">为普通用户创建专用密钥；普通用户只能进入画图页，不能查看设置和号池。</p>
                            </div>
                        </div>
                        <Button className="h-9 rounded-xl bg-stone-950 px-4 text-white hover:bg-stone-800"
                                onClick={() => setIsDialogOpen(true)}>
                            <Plus className="size-4"/>
                            创建用户密钥
                        </Button>
                    </div>

                    {revealedKey ? (
                        <div
                            className="space-y-4 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-4 text-sm text-emerald-900">
                            <div className="font-medium">新密钥仅展示一次，请立即保存：</div>
                            <div
                                className="mt-3 flex flex-col gap-3 rounded-lg border border-emerald-200 bg-white/80 p-3 md:flex-row md:items-center md:justify-between">
                                <code className="break-all font-mono text-[13px]">{revealedKey}</code>
                                <Button
                                    type="button"
                                    variant="outline"
                                    className="h-9 rounded-xl border-emerald-200 bg-white px-4 text-emerald-700"
                                    onClick={() => void handleCopy(revealedKey)}
                                >
                                    <Copy className="size-4"/>
                                    复制
                                </Button>
                            </div>
                            <div className="space-y-2">
                                <div className="flex items-center justify-between gap-3">
                                    <div>
                                        <div className="font-medium">调用演示</div>
                                        <p className="mt-1 text-xs leading-5 text-emerald-800/80">
                                            分发给客户时可附带这段 OpenAI 兼容 curl，Base URL 使用当前站点自动生成。
                                        </p>
                                    </div>
                                    <Button
                                        type="button"
                                        variant="outline"
                                        className="h-9 rounded-xl border-emerald-200 bg-white px-4 text-emerald-700"
                                        onClick={() => void handleCopy(revealedKeyDemoCurl)}
                                    >
                                        <Copy className="size-4"/>
                                        复制演示
                                    </Button>
                                </div>
                                <Textarea
                                    readOnly
                                    value={revealedKeyDemoCurl}
                                    className="min-h-56 resize-none rounded-xl border-emerald-200 bg-white/90 font-mono text-xs leading-5 text-stone-800 shadow-none"
                                />
                            </div>
                        </div>
                    ) : null}

                    {isLoading ? (
                        <div className="flex items-center justify-center py-10">
                            <LoaderCircle className="size-5 animate-spin text-stone-400"/>
                        </div>
                    ) : items.length === 0 ? (
                        <div className="rounded-xl bg-stone-50 px-6 py-10 text-center text-sm text-stone-500">
                            暂无普通用户密钥。点击右上角按钮后即可创建并分发给其他人。
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {items.map((item) => {
                                const isPending = pendingIds.has(item.id);
                                return (
                                    <div key={item.id}
                                         className="flex flex-col gap-3 rounded-xl border border-stone-200 bg-white px-4 py-4 md:flex-row md:items-center md:justify-between">
                                        <div className="min-w-0 space-y-2">
                                            <div className="flex flex-wrap items-center gap-2">
                                                <div
                                                    className="truncate text-sm font-medium text-stone-800">{item.name}</div>
                                                <Badge variant={item.enabled ? "success" : "secondary"}
                                                       className="rounded-md">
                                                    {item.enabled ? "已启用" : "已禁用"}
                                                </Badge>
                                            </div>
                                            <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-stone-500">
                                                <span>可用额度 {formatQuota(item.image_quota_available)}</span>
                                                <span>已预占 {formatQuota(item.image_quota_reserved)}</span>
                                                <span>创建时间 {formatDateTime(item.created_at)}</span>
                                                <span>最近使用 {formatDateTime(item.last_used_at)}</span>
                                            </div>
                                            <div className="break-all text-xs leading-5 text-stone-500">
                                                绑定 IP：{item.allowed_ips?.length ? item.allowed_ips.join("、") : "不限"}
                                            </div>
                                        </div>

                                        <div className="flex items-center gap-2">
                                            <Button
                                                type="button"
                                                variant="outline"
                                                className="h-9 rounded-xl border-stone-200 bg-white px-4 text-stone-700"
                                                onClick={() => openRechargeDialog(item)}
                                                disabled={isPending}
                                            >
                                                {isPending ? <LoaderCircle className="size-4 animate-spin"/> :
                                                    <Coins className="size-4"/>}
                                                充值
                                            </Button>
                                            <Button
                                                type="button"
                                                variant="outline"
                                                className="h-9 rounded-xl border-stone-200 bg-white px-4 text-stone-700"
                                                onClick={() => openEditDialog(item)}
                                                disabled={isPending}
                                            >
                                                {isPending ? <LoaderCircle className="size-4 animate-spin"/> :
                                                    <Pencil className="size-4"/>}
                                                编辑
                                            </Button>
                                            <Button
                                                type="button"
                                                variant="outline"
                                                className="h-9 rounded-xl border-stone-200 bg-white px-4 text-stone-700"
                                                onClick={() => void handleToggle(item)}
                                                disabled={isPending}
                                            >
                                                {isPending ? (
                                                    <LoaderCircle className="size-4 animate-spin"/>
                                                ) : item.enabled ? (
                                                    <Ban className="size-4"/>
                                                ) : (
                                                    <CheckCircle2 className="size-4"/>
                                                )}
                                                {item.enabled ? "禁用" : "启用"}
                                            </Button>
                                            <Button
                                                type="button"
                                                variant="outline"
                                                className="h-9 rounded-xl border-rose-200 bg-white px-4 text-rose-600 hover:bg-rose-50 hover:text-rose-700"
                                                onClick={() => setDeletingItem(item)}
                                                disabled={isPending}
                                            >
                                                {isPending ? <LoaderCircle className="size-4 animate-spin"/> :
                                                    <Trash2 className="size-4"/>}
                                                删除
                                            </Button>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </CardContent>
            </Card>

            <Dialog
                open={isDialogOpen}
                onOpenChange={(open) => {
                    setIsDialogOpen(open);
                    if (!open) {
                        setAllowedIps("");
                    }
                }}
            >
                <DialogContent className="rounded-2xl p-6">
                    <DialogHeader className="gap-2">
                        <DialogTitle>创建用户密钥</DialogTitle>
                        <DialogDescription className="text-sm leading-6">
                            可选填写一个备注名称，方便区分不同使用者；创建后会生成一条只能查看一次的原始密钥。
                        </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-stone-700">名称（可选）</label>
                        <Input
                            value={name}
                            onChange={(event) => setName(event.target.value)}
                            placeholder="例如：设计同学 A、运营临时账号"
                            className="h-11 rounded-xl border-stone-200 bg-white"
                        />
                    </div>
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-stone-700">初始图片额度</label>
                        <Input
                            type="number"
                            min="0"
                            step="0.1"
                            value={imageQuota}
                            onChange={(event) => setImageQuota(event.target.value)}
                            placeholder="0"
                            className="h-11 rounded-xl border-stone-200 bg-white"
                        />
                    </div>
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-stone-700">绑定 IP（可选）</label>
                        <Textarea
                            value={allowedIps}
                            onChange={(event) => setAllowedIps(event.target.value)}
                            placeholder={"每行一个 IP 或 CIDR，例如：\n203.0.113.10\n203.0.113.0/24"}
                            className="min-h-28 rounded-xl border-stone-200 bg-white font-mono text-xs shadow-none"
                        />
                        <p className="text-xs leading-5 text-stone-500">
                            留空表示不限制来源 IP；填写后该用户密钥只能从匹配的 IP 调用接口。
                        </p>
                    </div>
                    <DialogFooter>
                        <Button
                            type="button"
                            variant="secondary"
                            className="h-10 rounded-xl bg-stone-100 px-5 text-stone-700 hover:bg-stone-200"
                            onClick={() => {
                                setIsDialogOpen(false);
                                setAllowedIps("");
                            }}
                            disabled={isCreating}
                        >
                            取消
                        </Button>
                        <Button
                            type="button"
                            className="h-10 rounded-xl bg-stone-950 px-5 text-white hover:bg-stone-800"
                            onClick={() => void handleCreate()}
                            disabled={isCreating}
                        >
                            {isCreating ? <LoaderCircle className="size-4 animate-spin"/> : <Plus className="size-4"/>}
                            创建
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            <Dialog open={Boolean(deletingItem)} onOpenChange={(open) => (!open ? setDeletingItem(null) : null)}>
                <DialogContent className="rounded-2xl p-6">
                    <DialogHeader className="gap-2">
                        <DialogTitle>删除用户密钥</DialogTitle>
                        <DialogDescription className="text-sm leading-6">
                            确认删除用户密钥「{deletingItem?.name}」吗？删除后该密钥将无法继续调用接口。
                        </DialogDescription>
                    </DialogHeader>
                    <DialogFooter>
                        <Button
                            type="button"
                            variant="secondary"
                            className="h-10 rounded-xl bg-stone-100 px-5 text-stone-700 hover:bg-stone-200"
                            onClick={() => setDeletingItem(null)}
                            disabled={deletingItem ? pendingIds.has(deletingItem.id) : false}
                        >
                            取消
                        </Button>
                        <Button
                            type="button"
                            className="h-10 rounded-xl bg-rose-600 px-5 text-white hover:bg-rose-700"
                            onClick={() => void handleDelete()}
                            disabled={deletingItem ? pendingIds.has(deletingItem.id) : false}
                        >
                            {deletingItem && pendingIds.has(deletingItem.id) ?
                                <LoaderCircle className="size-4 animate-spin"/> : <Trash2 className="size-4"/>}
                            删除
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            <Dialog
                open={Boolean(editingItem)}
                onOpenChange={(open) => {
                    if (!open) {
                        setEditingItem(null);
                        setEditKey("");
                        setEditAllowedIps("");
                    }
                }}
            >
                <DialogContent className="rounded-2xl p-6">
                    <DialogHeader className="gap-2">
                        <DialogTitle>编辑用户密钥</DialogTitle>
                        <DialogDescription className="text-sm leading-6">
                            可以修改备注名称；如需更换专用密钥，直接填写新的原始密钥即可。留空则保持当前密钥不变。
                        </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4">
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-stone-700">名称</label>
                            <Input
                                value={editName}
                                onChange={(event) => setEditName(event.target.value)}
                                placeholder="例如：设计同学 A、运营临时账号"
                                className="h-11 rounded-xl border-stone-200 bg-white"
                            />
                        </div>
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-stone-700">新的专用密钥（可选）</label>
                            <Input
                                value={editKey}
                                onChange={(event) => setEditKey(event.target.value)}
                                placeholder="例如：sk-your-custom-user-key"
                                className="h-11 rounded-xl border-stone-200 bg-white font-mono"
                            />
                            <p className="text-xs leading-5 text-stone-500">
                                保存后旧密钥会立即失效，新密钥生效。系统仍只保存哈希，不会回显当前密钥。
                            </p>
                        </div>
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-stone-700">绑定 IP（可选）</label>
                            <Textarea
                                value={editAllowedIps}
                                onChange={(event) => setEditAllowedIps(event.target.value)}
                                placeholder={"每行一个 IP 或 CIDR，例如：\n203.0.113.10\n203.0.113.0/24"}
                                className="min-h-28 rounded-xl border-stone-200 bg-white font-mono text-xs shadow-none"
                            />
                            <p className="text-xs leading-5 text-stone-500">
                                清空后不限制来源 IP；支持 IPv4、IPv6 和 CIDR 网段。
                            </p>
                        </div>
                    </div>
                    <DialogFooter>
                        <Button
                            type="button"
                            variant="secondary"
                            className="h-10 rounded-xl bg-stone-100 px-5 text-stone-700 hover:bg-stone-200"
                            onClick={() => {
                                setEditingItem(null);
                                setEditKey("");
                                setEditAllowedIps("");
                            }}
                            disabled={editingItem ? pendingIds.has(editingItem.id) : false}
                        >
                            取消
                        </Button>
                        <Button
                            type="button"
                            className="h-10 rounded-xl bg-stone-950 px-5 text-white hover:bg-stone-800"
                            onClick={() => void handleEdit()}
                            disabled={editingItem ? pendingIds.has(editingItem.id) : false}
                        >
                            {editingItem && pendingIds.has(editingItem.id) ?
                                <LoaderCircle className="size-4 animate-spin"/> : <Pencil className="size-4"/>}
                            保存
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            <Dialog
                open={Boolean(rechargingItem)}
                onOpenChange={(open) => {
                    if (!open) {
                        setRechargingItem(null);
                        setRechargeAmount("");
                    }
                }}
            >
                <DialogContent className="rounded-2xl p-6">
                    <DialogHeader className="gap-2">
                        <DialogTitle>充值图片额度</DialogTitle>
                        <DialogDescription className="text-sm leading-6">
                            为用户密钥「{rechargingItem?.name}」追加图片额度；当前可用 {formatQuota(rechargingItem?.image_quota_available)}，已预占 {formatQuota(rechargingItem?.image_quota_reserved)}。
                        </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-stone-700">充值额度</label>
                        <Input
                            type="number"
                            min="0"
                            step="0.1"
                            value={rechargeAmount}
                            onChange={(event) => setRechargeAmount(event.target.value)}
                            placeholder="例如：10"
                            className="h-11 rounded-xl border-stone-200 bg-white"
                        />
                    </div>
                    <DialogFooter>
                        <Button
                            type="button"
                            variant="secondary"
                            className="h-10 rounded-xl bg-stone-100 px-5 text-stone-700 hover:bg-stone-200"
                            onClick={() => {
                                setRechargingItem(null);
                                setRechargeAmount("");
                            }}
                            disabled={isRecharging}
                        >
                            取消
                        </Button>
                        <Button
                            type="button"
                            className="h-10 rounded-xl bg-stone-950 px-5 text-white hover:bg-stone-800"
                            onClick={() => void handleRecharge()}
                            disabled={isRecharging}
                        >
                            {isRecharging ? <LoaderCircle className="size-4 animate-spin"/> :
                                <Coins className="size-4"/>}
                            充值
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>
        </>
    );
}

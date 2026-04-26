export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...options,
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        ...(options.headers ?? {}),
      },
    })
  } catch (err) {
    // fetch() rejects on network failure / CORS / DNS, NOT on 4xx/5xx.
    // Surface a friendly message instead of leaking "TypeError: Failed to fetch"
    // through every screen's error banner.
    const detail = err instanceof Error ? err.message : "Network error"
    throw new Error(`Network error — could not reach the server (${detail}).`)
  }

  // 204 No Content has no body — don't try to parse it.
  if (response.status === 204) {
    if (!response.ok) {
      throw new Error("Request failed.")
    }
    return {} as T
  }

  const data = await response.json().catch(() => ({} as { detail?: string }))

  if (!response.ok) {
    const detail =
      typeof (data as { detail?: unknown }).detail === "string"
        ? (data as { detail: string }).detail
        : `Request failed (${response.status}).`
    throw new Error(detail)
  }

  return data as T
}

export function assetUrl(path?: string | null): string {
  if (!path) return ""
  if (typeof path !== "string") return ""
  if (path.startsWith("http://") || path.startsWith("https://")) return path
  return `${API_BASE}${path}`
}

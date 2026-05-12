/**
 * In Vite dev, `VITE_API_URL` is usually unset so fetches go to the same host:port as the
 * UI and `/api` is proxied to the FastAPI server (works from a phone on your LAN).
 */

export function getApiBase(): string {
  const v = import.meta.env.VITE_API_URL
  if (typeof v === 'string' && v.trim() !== '') {
    const base = v.replace(/\/$/, '')
    // If the app is opened from another device (phone on LAN), a localhost API base
    // points to that device itself and always fails. Fall back to same-origin proxy.
    if (typeof window !== 'undefined') {
      try {
        const apiHost = new URL(base).hostname
        const uiHost = window.location.hostname
        const isApiLoopback = apiHost === '127.0.0.1' || apiHost === 'localhost'
        const isUiLoopback = uiHost === '127.0.0.1' || uiHost === 'localhost'
        if (isApiLoopback && !isUiLoopback) {
          return ''
        }
      } catch {
        /* keep explicit base if parsing fails */
      }
    }
    return base
  }
  if (import.meta.env.DEV) {
    return ''
  }
  return 'http://127.0.0.1:8000'
}

/**
 * `new URL(path, base)` needs an absolute base; when using same-origin relative API, use
 * the current page origin.
 */
export function originForUrlConstructor(apiBase: string): string {
  const b = (apiBase || '').replace(/\/$/, '')
  if (b) return b
  if (typeof window !== 'undefined' && window.location?.origin) {
    return window.location.origin
  }
  return 'http://127.0.0.1:5173'
}

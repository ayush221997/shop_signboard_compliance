/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** If unset in dev, API calls use same origin + Vite proxy. */
  readonly VITE_API_URL?: string
  /** Google Maps: enable Maps Embed API for this key. Used for map preview; geolocation still uses the browser. */
  readonly VITE_GOOGLE_MAPS_API_KEY?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

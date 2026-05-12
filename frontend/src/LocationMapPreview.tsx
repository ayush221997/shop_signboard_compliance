const MAPS_KEY = (import.meta.env.VITE_GOOGLE_MAPS_API_KEY as string | undefined)?.trim()

/**
 * Google Maps **Embed** preview (requires Maps Embed API enabled for the key).
 * User position still comes from `navigator.geolocation` in `LocationPanel` — the key is only for the map.
 */
export function LocationMapPreview({
  lat,
  lon,
  className = '',
}: {
  lat: number
  lon: number
  className?: string
}) {
  if (!MAPS_KEY || !Number.isFinite(lat) || !Number.isFinite(lon)) {
    return null
  }
  const src = `https://www.google.com/maps/embed/v1/view?key=${encodeURIComponent(MAPS_KEY)}&center=${encodeURIComponent(
    String(lat),
  )},${encodeURIComponent(String(lon))}&zoom=16&maptype=roadmap`

  return (
    <div className={`location-map ${className}`.trim()}>
      <iframe
        title="Map near your location"
        className="location-map__frame"
        width="100%"
        height="200"
        loading="lazy"
        allowFullScreen
        referrerPolicy="no-referrer-when-downgrade"
        src={src}
      />
    </div>
  )
}

export function hasGoogleMapsKey(): boolean {
  return Boolean(MAPS_KEY)
}

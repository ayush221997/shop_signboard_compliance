import { useCallback, useEffect, useState } from 'react'

import { originForUrlConstructor } from './apiBase'
import { LocationMapPreview } from './LocationMapPreview'
import type { OcrResult, ResolvedLocation } from './ocrTypes'

const LOCATION_OPTS: PositionOptions = { enableHighAccuracy: false, timeout: 20_000, maximumAge: 60_000 }

type Step = 'idle' | 'detecting' | 'ask' | 'manual' | 'done' | 'unavailable'

export function LocationPanel({
  apiBase,
  auditId,
  ocrResult,
  onOcrPatched,
}: {
  apiBase: string
  auditId: string | null
  ocrResult: OcrResult | null
  onOcrPatched: (r: OcrResult) => void
}) {
  const [step, setStep] = useState<Step>('idle')
  const [error, setError] = useState<string | null>(null)
  const [states, setStates] = useState<{ code: string; name: string }[]>([])
  const [look, setLook] = useState<{
    city: string
    stateName: string | null
    stateCode: string | null
    lat: number
    lon: number
  } | null>(null)
  const [manualCode, setManualCode] = useState('')

  const resolved = ocrResult?.resolved_location
  const alreadySet = Boolean((resolved as ResolvedLocation | undefined)?.state_code)

  const loadStates = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/api/compliance/states`)
      const data: unknown = await res.json()
      if (res.ok && data && typeof data === 'object' && 'states' in data) {
        const s = (data as { states: { code: string; name: string }[] }).states
        if (Array.isArray(s)) setStates(s)
      }
    } catch { /* keep empty */ }
  }, [apiBase])

  useEffect(() => {
    void loadStates()
  }, [loadStates])

  useEffect(() => {
    if (!auditId) {
      setStep('idle')
      setLook(null)
      setError(null)
      return
    }
    if (alreadySet) {
      setStep('done')
      return
    }
    if (!('geolocation' in navigator)) {
      setStep('unavailable')
      return
    }
    setError(null)
    setStep('detecting')
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const lat = pos.coords.latitude
        const lon = pos.coords.longitude
        try {
          const u = new URL('/api/geolookup', originForUrlConstructor(apiBase))
          u.searchParams.set('latitude', String(lat))
          u.searchParams.set('longitude', String(lon))
          const r = await fetch(u.toString())
          const d: unknown = await r.json().catch(() => ({}))
          if (!r.ok) {
            setError(typeof (d as { detail?: string }).detail === 'string' ? (d as { detail: string }).detail : 'Geolookup failed')
            setStep('manual')
            return
          }
          const row = d as { city?: string; state_name?: string | null; state_code?: string | null }
          setLook({
            city: (row.city || 'this area').trim() || 'this area',
            stateName: row.state_name ?? null,
            stateCode: (row.state_code || '').toUpperCase() || null,
            lat,
            lon,
          })
          setStep('ask')
        } catch (e) {
          setError(e instanceof Error ? e.message : 'Network error')
          setStep('manual')
        }
      },
      (err) => {
        setError(err.message || 'Position unavailable')
        setStep('manual')
      },
      LOCATION_OPTS,
    )
  }, [apiBase, auditId, alreadySet])

  const putAudit = useCallback(
    async (body: Record<string, unknown>) => {
      if (!auditId) return
      const res = await fetch(`${apiBase}/api/audit/${auditId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      const data: unknown = await res.json().catch(() => ({}))
      if (!res.ok) {
        const d = (data as { detail?: string }).detail
        throw new Error(typeof d === 'string' ? d : `Request failed (${res.status})`)
      }
      onOcrPatched(data as OcrResult)
      setStep('done')
    },
    [apiBase, auditId, onOcrPatched],
  )

  const onYes = async () => {
    if (!look) return
    if (!look.stateCode) {
      setError('Could not infer a supported Indian state/UT. Choose manually below.')
      setStep('manual')
      return
    }
    setError(null)
    try {
      await putAudit({
        latitude: look.lat,
        longitude: look.lon,
        state_code: look.stateCode,
        city: look.city,
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Update failed')
    }
  }

  const onSaveManual = async (e: React.FormEvent) => {
    e.preventDefault()
    const c = (manualCode || '').trim().toUpperCase()
    if (!c) {
      setError('Select a state/UT code.')
      return
    }
    setError(null)
    try {
      await putAudit({ state_code: c, city: look?.city || undefined })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Update failed')
    }
  }

  if (!auditId || !ocrResult) return null

  if (step === 'done' || alreadySet) {
    const rc = (resolved as ResolvedLocation | undefined)?.state_code || look?.stateCode
    const city = (resolved as ResolvedLocation | undefined)?.city
    const rlat = (resolved as ResolvedLocation | undefined)?.latitude
    const rlon = (resolved as ResolvedLocation | undefined)?.longitude
    const mapLat = typeof rlat === 'number' ? rlat : look?.lat
    const mapLon = typeof rlon === 'number' ? rlon : look?.lon
    return (
      <div className="location-panel location-panel--done" role="status">
        {mapLat != null && mapLon != null && (
          <LocationMapPreview lat={mapLat} lon={mapLon} />
        )}
        <p className="location-panel__p">
          Location:{' '}
          <strong>{rc || 'set'}</strong>
          {city ? ` — ${city}` : null}{' '}
          (used for threshed-language rules).
        </p>
      </div>
    )
  }

  if (step === 'idle' || step === 'detecting') {
    return (
      <div className="location-panel">
        <p className="location-panel__p">
          {step === 'detecting' ? 'Detecting your location for geo-based compliance…' : '…'}
        </p>
      </div>
    )
  }

  if (step === 'unavailable' || step === 'manual' || (step === 'ask' && error)) {
    return (
      <div className="location-panel">
        {error && <p className="message message--geo" role="status">{error}</p>}
        <h3 className="location-panel__h">Change location manually</h3>
        <p className="location-panel__disclaimer" role="note">
          Selecting the correct state is critical. Compliance rules vary significantly by jurisdiction.
        </p>
        <form className="location-panel__form" onSubmit={onSaveManual}>
          <label className="location-panel__label" htmlFor="st-sel">State/UT (ruleset)</label>
          <select
            id="st-sel"
            className="location-panel__select"
            value={manualCode}
            onChange={(e) => setManualCode(e.target.value)}
            required
          >
            <option value="">— Select —</option>
            {states.map((s) => (
              <option key={s.code} value={s.code}>
                {s.name} ({s.code})
              </option>
            ))}
          </select>
          <button type="submit" className="btn-audit-confirm location-panel__btn" disabled={!manualCode}>
            Save location
          </button>
        </form>
        {step === 'ask' && !error && (
          <p className="location-panel__hint">Or go back to the geolocation question if it reappears after a refresh.</p>
        )}
      </div>
    )
  }

  if (step === 'ask' && look) {
    return (
      <div className="location-panel" role="dialog" aria-label="Confirm signboard location">
        <LocationMapPreview lat={look.lat} lon={look.lon} />
        <p className="location-panel__disclaimer" role="note">
          Selecting the correct state is critical. Compliance rules vary significantly by jurisdiction.
        </p>
        <p className="location-panel__p">
          Signboard detected in <strong>{look.city}</strong>
          {look.stateName && (
            <>, {look.stateName}</>
          )}
          . Is this correct for compliance rules?
        </p>
        {error && <p className="message message--geo" role="alert">{error}</p>}
        <div className="location-panel__row">
          <button type="button" className="btn-audit-confirm" onClick={() => void onYes()}>
            Yes, use this location
          </button>
          <button
            type="button"
            className="btn-ghost"
            onClick={() => {
              setError(null)
              setStep('manual')
            }}
          >
            Change location manually
          </button>
        </div>
      </div>
    )
  }

  return null
}

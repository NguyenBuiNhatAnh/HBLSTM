import { useState } from "react"
import SymbolPage from "./pages/SymbolPage"

const SYMBOLS = ["DJI", "SPX", "NDX", "DXY", "NI225", "000001", "NIFTY", "UKX", "DEU40"]

export default function App() {
  const [active, setActive] = useState("DJI")

  return (
    <div style={{ fontFamily: "sans-serif", padding: "16px 24px" }}>
      {/* Tab navigation */}
      <div style={{ display: "flex", gap: 8, marginBottom: 24, flexWrap: "wrap" }}>
        {SYMBOLS.map(s => (
          <button key={s} onClick={() => setActive(s)} style={{
            padding: "6px 14px",
            borderRadius: 8,
            border: "none",
            cursor: "pointer",
            fontWeight: active === s ? 600 : 400,
            background: active === s ? "#6366f1" : "#f1f5f9",
            color: active === s ? "white" : "#374151",
          }}>
            {s}
          </button>
        ))}
      </div>

      <SymbolPage symbol={active} />
    </div>
  )
}
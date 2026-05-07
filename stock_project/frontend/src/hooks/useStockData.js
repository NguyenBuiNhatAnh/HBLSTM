import { useState, useEffect } from "react"

const MODELS = ["HBLSTM", "KNN", "DecisionTree", "LinearRegression"]

export function useStockData(symbol) {
  const [data, setData] = useState({})  // { HBLSTM: [...], KNN: [...], ... }
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetch_data = async () => {
      try {
        const res = await fetch(`/predictions/?symbol=${symbol}&limit=100`)
        const json = await res.json()

        // Group theo model_type
        const grouped = {}
        MODELS.forEach(m => {
          grouped[m] = json
            .filter(d => d.model_type === m)
            .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
        })
        setData(grouped)
      } catch (e) {
        console.error(e)
      } finally {
        setLoading(false)
      }
    }

    fetch_data()
    const interval = setInterval(fetch_data, 3000)
    return () => clearInterval(interval)
  }, [symbol])

  return { data, loading }
}
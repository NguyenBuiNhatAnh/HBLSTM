import ModelChart from "../components/ModelChart"
import { useStockData } from "../hooks/useStockData"

const MODELS = ["HBLSTM", "KNN", "DecisionTree", "LinearRegression"]

export default function SymbolPage({ symbol }) {
  const { data, loading } = useStockData(symbol)

  if (loading) return <p>Loading {symbol}...</p>

  return (
    <div>
      <h2 style={{ marginBottom: 16 }}>{symbol}</h2>
      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: 16,
      }}>
        {MODELS.map(model => (
          <ModelChart key={model} model={model} records={data[model] || []} />
        ))}
      </div>
    </div>
  )
}
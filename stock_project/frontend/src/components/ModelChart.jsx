import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts"
import { calcMetrics } from "../utils/metrics"

const MODEL_COLORS = {
  HBLSTM: "#6366f1",
  KNN: "#10b981",
  DecisionTree: "#f59e0b",
  LinearRegression: "#ef4444",
}

export default function ModelChart({ model, records }) {
  const metrics = calcMetrics(records)

  const chartData = records.map(r => ({
    time: new Date(r.timestamp).toLocaleTimeString(),
    predicted: parseFloat(r.predicted_price.toFixed(2)),
    actual: parseFloat(r.actual_price.toFixed(2)),
  }))

  return (
    <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <h3 style={{ margin: 0, fontSize: 15, fontWeight: 500 }}>{model}</h3>
        <div style={{ display: "flex", gap: 12, fontSize: 12, color: "#6b7280" }}>
          <span>RMSE: <b>{metrics.rmse}</b></span>
          <span>MSE: <b>{metrics.mse}</b></span>
          <span>MAPE: <b>{metrics.mape}%</b></span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={chartData}>
          <XAxis dataKey="time" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 10 }} domain={["auto", "auto"]} />
          <Tooltip />
          <Legend />
          <Line
            type="monotone" dataKey="actual"
            stroke="#94a3b8" dot={false} strokeWidth={1.5} name="Actual"
          />
          <Line
            type="monotone" dataKey="predicted"
            stroke={MODEL_COLORS[model]} dot={false} strokeWidth={1.5} name="Predicted"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
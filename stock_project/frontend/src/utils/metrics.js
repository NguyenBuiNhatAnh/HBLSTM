export function calcMetrics(records) {
  if (!records?.length) return { rmse: 0, mse: 0, mape: 0 }

  const n = records.length
  const mse = records.reduce((sum, r) => {
    return sum + Math.pow(r.predicted_price - r.actual_price, 2)
  }, 0) / n

  const mape = records.reduce((sum, r) => {
    return sum + Math.abs((r.predicted_price - r.actual_price) / r.actual_price)
  }, 0) / n * 100

  return {
    mse: mse.toFixed(4),
    rmse: Math.sqrt(mse).toFixed(4),
    mape: mape.toFixed(2),
  }
}
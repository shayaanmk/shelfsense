import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

// Merges real historical sales and the model's forecast into one series,
// keyed by date, so the chart shows "actual" trailing into "predicted"
// without a visual seam.
export default function ForecastChart({ sales, forecast }) {
  const byDate = new Map();

  (sales?.sales ?? []).forEach((row) => {
    byDate.set(row.date, { date: row.date, actual: row.units });
  });
  (forecast?.forecast ?? []).forEach((row) => {
    const existing = byDate.get(row.date) ?? { date: row.date };
    existing.predicted = row.predicted_units;
    byDate.set(row.date, existing);
  });

  const data = Array.from(byDate.values()).sort((a, b) => (a.date < b.date ? -1 : 1));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
        <XAxis dataKey="date" tick={{ fontSize: 11 }} />
        <YAxis tick={{ fontSize: 11 }} />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="actual" name="Actual units" stroke="#2563eb" dot={false} strokeWidth={2} />
        <Line
          type="monotone"
          dataKey="predicted"
          name={`Forecast (${forecast?.model ?? "model"})`}
          stroke="#dc2626"
          strokeDasharray="5 4"
          dot={false}
          strokeWidth={2}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

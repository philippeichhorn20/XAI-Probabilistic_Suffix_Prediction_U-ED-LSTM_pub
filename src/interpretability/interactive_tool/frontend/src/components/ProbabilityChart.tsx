import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { TopKPrediction } from "../api/types";

interface Props {
  topK: TopKPrediction[];
}

export default function ProbabilityChart({ topK }: Props) {
  const data = topK.map((t) => ({
    name: t.activity.length > 25 ? t.activity.slice(0, 23) + ".." : t.activity,
    probability: +(t.probability * 100).toFixed(1),
  }));

  return (
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={data} layout="vertical" margin={{ left: 10, right: 20 }}>
        <XAxis type="number" domain={[0, 100]} unit="%" tick={{ fontSize: 12 }} />
        <YAxis
          dataKey="name"
          type="category"
          width={150}
          tick={{ fontSize: 12 }}
        />
        <Tooltip formatter={(v: number) => `${v}%`} />
        <Bar dataKey="probability" radius={[0, 4, 4, 0]}>
          {data.map((_, i) => (
            <Cell key={i} fill={i === 0 ? "#1f77b4" : "#90caf9"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

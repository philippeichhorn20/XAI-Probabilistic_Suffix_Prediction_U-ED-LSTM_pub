import { useMemo } from "react";
import { ResponsiveHeatMap } from "@nivo/heatmap";
import type { AttributionResponse } from "../api/types";

interface Props {
  data: AttributionResponse;
  title: string;
  midpoint?: number;
}

export default function AttributionHeatmap({ data, title, midpoint }: Props) {
  const { nivoData, minVal, maxVal } = useMemo(() => {
    let min = Infinity;
    let max = -Infinity;

    // Build Nivo-style data: each row = one feature, each cell = one step
    const rows = data.feature_names.map((feat) => {
      const vals = data.attributions[feat] ?? [];
      const cells: Record<string, number> = {};
      data.step_labels.forEach((label, j) => {
        const v = vals[j] ?? 0;
        cells[label] = v;
        if (v < min) min = v;
        if (v > max) max = v;
      });
      return { id: feat, data: data.step_labels.map((l) => ({ x: l, y: cells[l] })) };
    });

    return { nivoData: rows, minVal: min, maxVal: max };
  }, [data]);

  // Symmetric scale around midpoint (default 0)
  const mid = midpoint ?? 0;
  const absMax = Math.max(Math.abs(minVal - mid), Math.abs(maxVal - mid), 0.001);

  const height = Math.max(300, data.feature_names.length * 28 + 80);

  return (
    <div style={{ height }}>
      <div
        style={{
          textAlign: "center",
          fontWeight: 500,
          fontSize: 14,
          marginBottom: 4,
        }}
      >
        {title}
      </div>
      <ResponsiveHeatMap
        data={nivoData}
        margin={{ top: 10, right: 30, bottom: 60, left: 140 }}
        axisBottom={{
          tickRotation: -45,
          tickSize: 5,
          tickPadding: 5,
        }}
        axisLeft={{ tickSize: 5, tickPadding: 5 }}
        colors={{
          type: "diverging",
          scheme: "red_blue",
          divergeAt: 0.5,
          minValue: mid - absMax,
          maxValue: mid + absMax,
        }}
        emptyColor="#f5f5f5"
        borderWidth={1}
        borderColor="#e0e0e0"
        labelTextColor={{ from: "color", modifiers: [["darker", 3]] }}
        legends={[
          {
            anchor: "bottom",
            translateX: 0,
            translateY: 50,
            length: 200,
            thickness: 10,
            direction: "row",
            tickPosition: "after",
            tickSize: 3,
            tickSpacing: 4,
          },
        ]}
        hoverTarget="cell"
        tooltip={({ cell }) => (
          <div
            style={{
              background: "#fff",
              padding: "6px 10px",
              border: "1px solid #ccc",
              borderRadius: 4,
              fontSize: 12,
            }}
          >
            <strong>{cell.serieId}</strong> @ {String(cell.data.x)}
            <br />
            Attribution: {Number(cell.data.y).toFixed(4)}
          </div>
        )}
      />
    </div>
  );
}

import { useMemo } from "react";
import { Box, Divider, Paper, Typography } from "@mui/material";
import { useCaseStore } from "../store/useCaseStore";
import AttributionHeatmap from "./AttributionHeatmap";
import type { AttributionResponse } from "../api/types";

export default function AttributionPanel() {
  const { originalAttribution, modifiedAttribution, showAttributions } =
    useCaseStore();

  if (!showAttributions) return null;
  if (!originalAttribution && !modifiedAttribution) return null;

  return (
    <Box>
      <Divider sx={{ my: 1 }} />
      <Typography variant="h6" gutterBottom>
        Attribution Analysis
      </Typography>

      <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
        {/* Original */}
        <Paper sx={{ flex: 1, minWidth: 400, p: 2 }}>
          <Typography variant="subtitle1" color="primary" gutterBottom>
            Original Case Attribution
          </Typography>
          {originalAttribution ? (
            <>
              <Typography variant="caption" color="text.secondary">
                {originalAttribution.target_description}
              </Typography>
              <AttributionHeatmap
                data={originalAttribution}
                title="Original Case"
              />
            </>
          ) : (
            <Typography variant="body2" color="text.secondary">
              No attribution computed
            </Typography>
          )}
        </Paper>

        {/* Modified */}
        <Paper sx={{ flex: 1, minWidth: 400, p: 2 }}>
          <Typography variant="subtitle1" color="secondary" gutterBottom>
            Modified Case Attribution
          </Typography>
          {modifiedAttribution ? (
            <>
              <Typography variant="caption" color="text.secondary">
                {modifiedAttribution.target_description}
              </Typography>
              <AttributionHeatmap
                data={modifiedAttribution}
                title="Modified Case"
              />
            </>
          ) : (
            <Typography variant="body2" color="text.secondary">
              No attribution computed
            </Typography>
          )}
        </Paper>
      </Box>

      {/* Difference heatmap */}
      {originalAttribution && modifiedAttribution && (
        <DifferenceHeatmap
          original={originalAttribution}
          modified={modifiedAttribution}
        />
      )}
    </Box>
  );
}

function DifferenceHeatmap({
  original,
  modified,
}: {
  original: AttributionResponse;
  modified: AttributionResponse;
}) {
  const diffData = useMemo((): AttributionResponse | null => {
    // Must have same step labels to produce a meaningful diff
    if (original.step_labels.length !== modified.step_labels.length) return null;

    const attributions: Record<string, number[]> = {};
    for (const feat of original.feature_names) {
      const origVals = original.attributions[feat] ?? [];
      const modVals = modified.attributions[feat] ?? [];
      attributions[feat] = origVals.map((v, i) => (modVals[i] ?? 0) - v);
    }

    return {
      attributions,
      feature_names: original.feature_names,
      step_labels: original.step_labels,
      target_description: "Attribution Difference (Modified - Original)",
      convergence_delta: null,
      prefix_length: original.prefix_length,
    };
  }, [original, modified]);

  if (!diffData) return null;

  return (
    <Paper sx={{ mt: 2, p: 2 }}>
      <Typography variant="subtitle1" gutterBottom>
        Attribution Change (Modified - Original)
      </Typography>
      <AttributionHeatmap data={diffData} title="Difference" midpoint={0} />
    </Paper>
  );
}

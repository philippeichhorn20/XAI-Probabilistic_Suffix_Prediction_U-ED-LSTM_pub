import { Alert, Box, Divider, Paper, Typography } from "@mui/material";
import { useCaseStore } from "../store/useCaseStore";
import ProbabilityChart from "./ProbabilityChart";

export default function PredictionPanel() {
  const { originalPrediction, modifiedPrediction } = useCaseStore();

  if (!originalPrediction && !modifiedPrediction) return null;

  const changed =
    originalPrediction &&
    modifiedPrediction &&
    originalPrediction.predicted_sequence !== modifiedPrediction.predicted_sequence;

  return (
    <Box>
      <Divider sx={{ my: 1 }} />
      <Typography variant="h6" gutterBottom>
        Prediction Results
      </Typography>

      <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
        {/* Original */}
        <Paper sx={{ flex: 1, minWidth: 300, p: 2 }}>
          <Typography variant="subtitle1" color="primary" gutterBottom>
            Original Case
          </Typography>
          {originalPrediction && (
            <>
              <Typography variant="body2" sx={{ mb: 1 }}>
                <b>Predicted:</b> {originalPrediction.predicted_sequence}
              </Typography>
              {originalPrediction.actual_suffix_activities && (
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <b>Actual:</b>{" "}
                  {originalPrediction.actual_suffix_activities.join(" -> ")}
                </Typography>
              )}
              <Typography variant="caption" color="text.secondary">
                Top predictions (Step 1)
              </Typography>
              {originalPrediction.steps.length > 0 && (
                <ProbabilityChart topK={originalPrediction.steps[0].top_k} />
              )}
            </>
          )}
        </Paper>

        {/* Modified */}
        <Paper sx={{ flex: 1, minWidth: 300, p: 2 }}>
          <Typography variant="subtitle1" color="secondary" gutterBottom>
            Modified Case
          </Typography>
          {modifiedPrediction && (
            <>
              <Typography variant="body2" sx={{ mb: 1 }}>
                <b>Predicted:</b> {modifiedPrediction.predicted_sequence}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Top predictions (Step 1)
              </Typography>
              {modifiedPrediction.steps.length > 0 && (
                <ProbabilityChart topK={modifiedPrediction.steps[0].top_k} />
              )}
            </>
          )}
        </Paper>
      </Box>

      {changed !== undefined && (
        <Alert severity={changed ? "success" : "info"} sx={{ mt: 2 }}>
          {changed
            ? "The modification changed the prediction!"
            : "The prediction remained the same despite the modification."}
        </Alert>
      )}
    </Box>
  );
}

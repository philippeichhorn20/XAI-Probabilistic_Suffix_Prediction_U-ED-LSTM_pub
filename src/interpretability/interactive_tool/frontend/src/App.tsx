import { useEffect, useState, useCallback } from "react";
import {
  Box,
  Button,
  Checkbox,
  CircularProgress,
  FormControlLabel,
  Slider,
  Tab,
  Tabs,
  Typography,
} from "@mui/material";
import Layout from "./components/Layout";
import Sidebar from "./components/Sidebar";
import EventSequence from "./components/EventSequence";
import CaseLevelEditor from "./components/CaseLevelEditor";
import PredictionPanel from "./components/PredictionPanel";
import AttributionPanel from "./components/AttributionPanel";
import EventEditDialog from "./components/EventEditDialog";
import { useCaseStore } from "./store/useCaseStore";

function PredictionControls() {
  const {
    originalCase,
    prefixLength,
    setPrefixLength,
    showAttributions,
    setShowAttributions,
    runPrediction,
    runAttribution,
    loadingPrediction,
    loadingAttribution,
  } = useCaseStore();

  const handleRun = useCallback(async () => {
    await runPrediction();
    if (showAttributions) {
      await runAttribution();
    }
  }, [runPrediction, runAttribution, showAttributions]);

  const maxPrefix = originalCase
    ? Math.max(1, originalCase.events.length - 1)
    : 1;

  if (!originalCase) return null;

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        gap: 3,
        flexWrap: "wrap",
      }}
    >
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, minWidth: 250 }}>
        <Typography variant="body2" sx={{ whiteSpace: "nowrap" }}>
          Prefix Length: {prefixLength}
        </Typography>
        <Slider
          value={prefixLength}
          min={1}
          max={maxPrefix}
          onChange={(_, v) => setPrefixLength(v as number)}
          valueLabelDisplay="auto"
          size="small"
          sx={{ minWidth: 120 }}
        />
      </Box>

      <FormControlLabel
        control={
          <Checkbox
            checked={showAttributions}
            onChange={(_, v) => setShowAttributions(v)}
            size="small"
          />
        }
        label="Compute Attributions"
      />

      <Button
        variant="contained"
        color="primary"
        onClick={handleRun}
        disabled={loadingPrediction || loadingAttribution}
        startIcon={
          loadingPrediction || loadingAttribution ? (
            <CircularProgress size={16} />
          ) : undefined
        }
      >
        {loadingPrediction
          ? "Predicting..."
          : loadingAttribution
            ? "Computing Attributions..."
            : "Run Prediction"}
      </Button>
    </Box>
  );
}

function CaseExplorerTab() {
  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
      <CaseLevelEditor />
      <EventSequence />
      <PredictionControls />
      <PredictionPanel />
      <AttributionPanel />
    </Box>
  );
}

function PlaceholderTab() {
  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: "40vh",
        color: "text.secondary",
        fontSize: "1.1rem",
      }}
    >
      This tab is not yet configured.
    </Box>
  );
}

export default function App() {
  const loadModels = useCaseStore((s) => s.loadModels);
  const originalCase = useCaseStore((s) => s.originalCase);
  const [tab, setTab] = useState(0);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  return (
    <Layout sidebar={<Sidebar />}>
      {originalCase ? (
        <Box>
          <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 2 }}>
            <Tab label="Case Explorer" />
            <Tab label="Tab 2" />
          </Tabs>

          {tab === 0 && <CaseExplorerTab />}
          {tab === 1 && <PlaceholderTab />}
        </Box>
      ) : (
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            height: "60vh",
            color: "text.secondary",
            fontSize: "1.1rem",
          }}
        >
          Select a model and load a case from the sidebar to get started.
        </Box>
      )}
      <EventEditDialog />
    </Layout>
  );
}

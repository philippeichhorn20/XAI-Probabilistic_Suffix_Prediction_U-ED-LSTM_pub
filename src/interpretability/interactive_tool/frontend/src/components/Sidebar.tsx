import { useEffect, useState, useCallback } from "react";
import {
  Box,
  Button,
  Divider,
  MenuItem,
  TextField,
  Typography,
  CircularProgress,
} from "@mui/material";
import { useCaseStore } from "../store/useCaseStore";

export default function Sidebar() {
  const {
    models,
    selectedModel,
    setSelectedModel,
    caseNames,
    casesTotal,
    searchCases,
    loadCaseData,
    loadingCase,
  } = useCaseStore();

  const [localCase, setLocalCase] = useState("");
  const [localSearch, setLocalSearch] = useState("");

  // Fetch cases when model changes
  useEffect(() => {
    if (selectedModel) {
      searchCases("");
    }
  }, [selectedModel, searchCases]);

  // Debounced search
  useEffect(() => {
    const t = setTimeout(() => searchCases(localSearch), 300);
    return () => clearTimeout(t);
  }, [localSearch, searchCases]);

  const handleLoad = useCallback(() => {
    if (localCase) loadCaseData(localCase);
  }, [localCase, loadCaseData]);

  return (
    <Box sx={{ p: 2, display: "flex", flexDirection: "column", gap: 2 }}>
      <Typography variant="subtitle2" color="text.secondary">
        Model & Case Selection
      </Typography>

      {/* Model select */}
      <TextField
        select
        label="Model"
        size="small"
        value={selectedModel}
        onChange={(e) => {
          setSelectedModel(e.target.value);
          setLocalCase("");
          setLocalSearch("");
        }}
      >
        {models.map((m) => (
          <MenuItem key={m.name} value={m.name}>
            {m.display_name}
          </MenuItem>
        ))}
      </TextField>

      <Divider />

      {/* Case search + select */}
      {selectedModel && (
        <>
          <TextField
            label="Search cases"
            size="small"
            value={localSearch}
            onChange={(e) => setLocalSearch(e.target.value)}
          />
          <TextField
            select
            label={`Case (${casesTotal} total)`}
            size="small"
            value={localCase}
            onChange={(e) => setLocalCase(e.target.value)}
          >
            {caseNames.map((c) => (
              <MenuItem key={c} value={c}>
                {c}
              </MenuItem>
            ))}
          </TextField>
          <Button
            variant="contained"
            onClick={handleLoad}
            disabled={!localCase || loadingCase}
            startIcon={loadingCase ? <CircularProgress size={16} /> : undefined}
          >
            {loadingCase ? "Loading..." : "Load Case"}
          </Button>
        </>
      )}
    </Box>
  );
}

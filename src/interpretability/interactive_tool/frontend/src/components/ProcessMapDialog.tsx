import { useState, useCallback } from "react";
import {
  Box,
  Button,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
  LinearProgress,
  Typography,
} from "@mui/material";
import CloseIcon from "@mui/icons-material/Close";
import { fetchProcessMap } from "../api/client";
import { useCaseStore } from "../store/useCaseStore";

export default function ProcessMapDialog() {
  const { selectedModel, selectedCase, prefixLength } = useCaseStore();

  const [open, setOpen] = useState(false);
  const [svg, setSvg] = useState("");
  const [loading, setLoading] = useState(false);

  const handleOpen = useCallback(async () => {
    if (!selectedModel || !selectedCase) return;
    setOpen(true);
    setLoading(true);
    try {
      const result = await fetchProcessMap(
        selectedModel,
        selectedCase,
        prefixLength,
      );
      setSvg(result);
    } catch {
      setSvg("<p>Failed to load process map.</p>");
    } finally {
      setLoading(false);
    }
  }, [selectedModel, selectedCase, prefixLength]);

  return (
    <>
      <Button size="small" variant="outlined" onClick={handleOpen}>
        View in Process Map
      </Button>

      <Dialog
        open={open}
        onClose={() => setOpen(false)}
        maxWidth={false}
        fullWidth
        PaperProps={{ sx: { maxWidth: "95vw", maxHeight: "95vh" } }}
      >
        <DialogTitle sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          Process Performance Map
          <IconButton onClick={() => setOpen(false)} size="small">
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        {loading && <LinearProgress />}
        <DialogContent>
          {loading ? (
            <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", py: 8, gap: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Building process map (first load collects predictions across all test cases)...
              </Typography>
            </Box>
          ) : (
            <Box
              sx={{
                overflow: "auto",
                "& svg": { maxWidth: "100%", height: "auto" },
              }}
              dangerouslySetInnerHTML={{ __html: svg }}
            />
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}

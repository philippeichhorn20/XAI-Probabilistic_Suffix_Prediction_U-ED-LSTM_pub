import { useState, useEffect, useCallback } from "react";
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  MenuItem,
  TextField,
  Typography,
  Box,
} from "@mui/material";
import { useCaseStore } from "../store/useCaseStore";
import type { EventData } from "../api/types";

export default function EventEditDialog() {
  const {
    editingEventIdx,
    setEditingEventIdx,
    modifiedCase,
    originalCase,
    catInfo,
    numInfo,
    caseLevelCat,
    caseLevelNum,
    updateEvent,
  } = useCaseStore();

  const open = editingEventIdx !== null;
  const idx = editingEventIdx ?? 0;

  // Local draft of the event being edited
  const [draft, setDraft] = useState<EventData | null>(null);

  useEffect(() => {
    if (open && modifiedCase) {
      setDraft(JSON.parse(JSON.stringify(modifiedCase.events[idx])));
    }
  }, [open, idx, modifiedCase]);

  const handleClose = useCallback(() => setEditingEventIdx(null), [setEditingEventIdx]);

  const handleApply = useCallback(() => {
    if (draft !== null) {
      updateEvent(idx, draft);
    }
    handleClose();
  }, [draft, idx, updateEvent, handleClose]);

  const handleReset = useCallback(() => {
    if (originalCase) {
      setDraft(JSON.parse(JSON.stringify(originalCase.events[idx])));
    }
  }, [originalCase, idx]);

  if (!draft) return null;

  const eventCatInfo = catInfo.filter((c) => !caseLevelCat.includes(c.name));
  const eventNumInfo = numInfo.filter((n) => !caseLevelNum.includes(n.name));

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>Edit Event {idx + 1}</DialogTitle>
      <DialogContent dividers>
        {/* Categorical features */}
        {eventCatInfo.length > 0 && (
          <>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Categorical Features
            </Typography>
            {eventCatInfo.map((cat) => {
              const val = draft.categorical[cat.name] ?? 0;
              return (
                <TextField
                  key={cat.name}
                  select
                  label={cat.name}
                  size="small"
                  fullWidth
                  sx={{ mb: 1.5 }}
                  value={val}
                  onChange={(e) =>
                    setDraft((d) =>
                      d
                        ? {
                            ...d,
                            categorical: {
                              ...d.categorical,
                              [cat.name]: Number(e.target.value),
                            },
                          }
                        : d,
                    )
                  }
                >
                  {Array.from({ length: cat.num_values }, (_, i) => (
                    <MenuItem key={i} value={i}>
                      {(cat.idx_to_value[i] ?? `Unknown_${i}`).slice(0, 40)}
                    </MenuItem>
                  ))}
                </TextField>
              );
            })}
          </>
        )}

        {eventCatInfo.length > 0 && eventNumInfo.length > 0 && (
          <Divider sx={{ my: 2 }} />
        )}

        {/* Numerical features */}
        {eventNumInfo.length > 0 && (
          <>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Numerical Features
            </Typography>
            {eventNumInfo.map((num) => {
              const val = draft.numerical[num.name] ?? 0;
              return (
                <TextField
                  key={num.name}
                  label={num.name}
                  type="number"
                  size="small"
                  fullWidth
                  sx={{ mb: 1.5 }}
                  value={val}
                  onChange={(e) =>
                    setDraft((d) =>
                      d
                        ? {
                            ...d,
                            numerical: {
                              ...d.numerical,
                              [num.name]: Number(e.target.value),
                            },
                          }
                        : d,
                    )
                  }
                  inputProps={{ step: 0.0001 }}
                />
              );
            })}
          </>
        )}

        {eventCatInfo.length === 0 && eventNumInfo.length === 0 && (
          <Typography variant="body2" color="text.secondary">
            All features for this event are case-level. Edit them in the Case
            Attributes section.
          </Typography>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={handleReset}>Reset to Original</Button>
        <Button onClick={handleClose}>Cancel</Button>
        <Button variant="contained" onClick={handleApply}>
          Apply
        </Button>
      </DialogActions>
    </Dialog>
  );
}

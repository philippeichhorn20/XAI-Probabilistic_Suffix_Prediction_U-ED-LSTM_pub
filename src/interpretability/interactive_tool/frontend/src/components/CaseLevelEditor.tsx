import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  MenuItem,
  TextField,
  Typography,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { useCaseStore } from "../store/useCaseStore";

export default function CaseLevelEditor() {
  const {
    catInfo,
    numInfo,
    caseLevelCat,
    caseLevelNum,
    modifiedCase,
    originalCase,
    updateCaseLevelFeature,
  } = useCaseStore();

  const clCatInfo = catInfo.filter((c) => caseLevelCat.includes(c.name));
  const clNumInfo = numInfo.filter((n) => caseLevelNum.includes(n.name));

  if (clCatInfo.length === 0 && clNumInfo.length === 0) return null;
  if (!modifiedCase || !originalCase) return null;

  const total = clCatInfo.length + clNumInfo.length;

  return (
    <Accordion defaultExpanded>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Typography variant="subtitle1">
          Case Attributes ({total} constant features)
        </Typography>
      </AccordionSummary>
      <AccordionDetails>
        <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: "block" }}>
          These features have the same value for all events. Changes apply to
          every event.
        </Typography>

        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
          {clCatInfo.map((cat) => {
            const val = modifiedCase.events[0]?.categorical[cat.name] ?? 0;
            return (
              <TextField
                key={cat.name}
                select
                label={cat.name}
                size="small"
                sx={{ minWidth: 200 }}
                value={val}
                onChange={(e) =>
                  updateCaseLevelFeature(
                    "categorical",
                    cat.name,
                    Number(e.target.value),
                  )
                }
              >
                {Array.from({ length: cat.num_values }, (_, i) => (
                  <MenuItem key={i} value={i}>
                    {(cat.idx_to_value[i] ?? `Unknown_${i}`).slice(0, 30)}
                  </MenuItem>
                ))}
              </TextField>
            );
          })}

          {clNumInfo.map((num) => {
            const val = modifiedCase.events[0]?.numerical[num.name] ?? 0;
            return (
              <TextField
                key={num.name}
                label={num.name}
                type="number"
                size="small"
                sx={{ minWidth: 200 }}
                value={val}
                onChange={(e) =>
                  updateCaseLevelFeature(
                    "numerical",
                    num.name,
                    Number(e.target.value),
                  )
                }
                inputProps={{ step: 0.0001 }}
              />
            );
          })}
        </Box>
      </AccordionDetails>
    </Accordion>
  );
}

import { Box, Button, Paper, Typography } from "@mui/material";
import { useCaseStore } from "../store/useCaseStore";
import { chipColors } from "../theme";
import ProcessMapDialog from "./ProcessMapDialog";

function getValueLabel(
  featureName: string,
  value: number,
  catInfo: { name: string; idx_to_value: Record<number, string> }[],
): string {
  const info = catInfo.find((c) => c.name === featureName);
  if (info) {
    const label = info.idx_to_value[value] ?? `${value}`;
    return label.length > 22 ? label.slice(0, 20) + ".." : label;
  }
  // numerical — show up to 4 decimals, strip trailing zeros
  return Number(value).toFixed(4).replace(/\.?0+$/, "");
}

function isEventModified(
  orig: { categorical: Record<string, number>; numerical: Record<string, number> },
  mod: { categorical: Record<string, number>; numerical: Record<string, number> },
): boolean {
  for (const k of Object.keys(orig.categorical)) {
    if (orig.categorical[k] !== mod.categorical[k]) return true;
  }
  for (const k of Object.keys(orig.numerical)) {
    if (Math.abs((orig.numerical[k] ?? 0) - (mod.numerical[k] ?? 0)) > 1e-6)
      return true;
  }
  return false;
}

function isFeatureModified(
  featureName: string,
  type: "categorical" | "numerical",
  orig: { categorical: Record<string, number>; numerical: Record<string, number> },
  mod: { categorical: Record<string, number>; numerical: Record<string, number> },
): boolean {
  if (type === "categorical") {
    return orig.categorical[featureName] !== mod.categorical[featureName];
  }
  return (
    Math.abs(
      (orig.numerical[featureName] ?? 0) - (mod.numerical[featureName] ?? 0),
    ) > 1e-6
  );
}

export default function EventSequence() {
  const {
    originalCase,
    modifiedCase,
    prefixLength,
    conceptName,
    catInfo,
    numInfo,
    caseLevelCat,
    caseLevelNum,
    setEditingEventIdx,
    resetCase,
  } = useCaseStore();

  if (!originalCase || !modifiedCase) return null;

  // Per-event features (exclude activity + case-level)
  const eventCatFeatures = catInfo.filter(
    (c) => c.name !== conceptName && !caseLevelCat.includes(c.name),
  );
  const eventNumFeatures = numInfo.filter(
    (n) => !caseLevelNum.includes(n.name),
  );

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Activity Sequence
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
        Click on any event to edit it
      </Typography>

      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1.5 }}>
        {modifiedCase.events.map((ev, i) => {
          const origEv = originalCase.events[i];
          const actIdx = ev.categorical[conceptName] ?? 0;
          const origActIdx = origEv.categorical[conceptName] ?? 0;
          const actInfo = catInfo.find((c) => c.name === conceptName);
          const actLabel = actInfo?.idx_to_value[actIdx] ?? `Unknown_${actIdx}`;
          const origActLabel = actInfo?.idx_to_value[origActIdx] ?? `Unknown_${origActIdx}`;
          const activityChanged = actIdx !== origActIdx;
          const modified = isEventModified(origEv, ev);
          const isPrefix = i < prefixLength;

          const bg = isPrefix ? chipColors.prefix : chipColors.suffix;

          return (
            <Paper
              key={i}
              elevation={modified ? 3 : 1}
              onClick={() => setEditingEventIdx(i)}
              sx={{
                cursor: "pointer",
                minWidth: 140,
                maxWidth: 200,
                borderTop: `3px solid ${bg}`,
                p: 1,
                "&:hover": {
                  boxShadow: 4,
                  transform: "translateY(-2px)",
                },
                transition: "transform 0.15s, box-shadow 0.15s",
              }}
            >
              {/* Activity (prominent) */}
              {activityChanged ? (
                <Box sx={{ mb: 0.5 }}>
                  <Typography
                    variant="body2"
                    sx={{
                      fontWeight: 400,
                      color: "text.disabled",
                      textDecoration: "line-through",
                      fontSize: "0.78rem",
                      lineHeight: 1.2,
                      wordBreak: "break-word",
                    }}
                  >
                    {i + 1}.{" "}
                    {origActLabel.length > 20
                      ? origActLabel.slice(0, 18) + ".."
                      : origActLabel}
                  </Typography>
                  <Typography
                    variant="body2"
                    sx={{
                      fontWeight: 600,
                      color: chipColors.modified,
                      lineHeight: 1.2,
                      wordBreak: "break-word",
                    }}
                  >
                    {actLabel.length > 20
                      ? actLabel.slice(0, 18) + ".."
                      : actLabel}
                  </Typography>
                </Box>
              ) : (
                <Typography
                  variant="body2"
                  sx={{
                    fontWeight: 600,
                    color: bg,
                    mb: 0.5,
                    lineHeight: 1.2,
                    wordBreak: "break-word",
                  }}
                >
                  {i + 1}.{" "}
                  {actLabel.length > 20
                    ? actLabel.slice(0, 18) + ".."
                    : actLabel}
                </Typography>
              )}

              {/* Other per-event features (subdued) */}
              {(eventCatFeatures.length > 0 || eventNumFeatures.length > 0) && (
                <Box
                  sx={{
                    mt: 0.5,
                    display: "flex",
                    flexDirection: "column",
                    gap: "2px",
                  }}
                >
                  {eventCatFeatures.map((feat) => {
                    const val = ev.categorical[feat.name] ?? 0;
                    const origVal = origEv.categorical[feat.name] ?? 0;
                    const featModified = isFeatureModified(
                      feat.name,
                      "categorical",
                      origEv,
                      ev,
                    );
                    return (
                      <FeatureRow
                        key={feat.name}
                        name={feat.name}
                        value={getValueLabel(feat.name, val, catInfo)}
                        modified={featModified}
                        originalValue={
                          featModified
                            ? getValueLabel(feat.name, origVal, catInfo)
                            : undefined
                        }
                      />
                    );
                  })}
                  {eventNumFeatures.map((feat) => {
                    const val = ev.numerical[feat.name] ?? 0;
                    const origVal = origEv.numerical[feat.name] ?? 0;
                    const featModified = isFeatureModified(
                      feat.name,
                      "numerical",
                      origEv,
                      ev,
                    );
                    return (
                      <FeatureRow
                        key={feat.name}
                        name={feat.name}
                        value={getValueLabel(feat.name, val, catInfo)}
                        modified={featModified}
                        originalValue={
                          featModified
                            ? getValueLabel(feat.name, origVal, catInfo)
                            : undefined
                        }
                      />
                    );
                  })}
                </Box>
              )}
            </Paper>
          );
        })}
      </Box>

      {/* Legend + reset */}
      <Box
        sx={{
          display: "flex",
          gap: 3,
          mt: 2,
          alignItems: "center",
          flexWrap: "wrap",
        }}
      >
        <LegendDot color={chipColors.prefix} label="Prefix (input)" />
        <LegendDot color={chipColors.suffix} label="Suffix (ground truth)" />
        <LegendDot color={chipColors.modified} label="Modified" />
        <Button size="small" onClick={resetCase}>
          Reset All
        </Button>
        <ProcessMapDialog />
      </Box>
    </Box>
  );
}

function FeatureRow({
  name,
  value,
  modified,
  originalValue,
}: {
  name: string;
  value: string;
  modified: boolean;
  originalValue?: string;
}) {
  const shortName = name.length > 14 ? name.slice(0, 12) + ".." : name;
  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: "space-between",
        gap: 0.5,
        alignItems: "baseline",
      }}
    >
      <Typography
        variant="caption"
        sx={{
          color: modified ? chipColors.modified : "text.disabled",
          fontSize: "0.68rem",
          lineHeight: 1.3,
          flexShrink: 0,
        }}
      >
        {shortName}
      </Typography>

      {modified && originalValue != null ? (
        /* Show struck-through original + new value */
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-end",
            overflow: "hidden",
            minWidth: 0,
          }}
        >
          <Typography
            variant="caption"
            sx={{
              color: "text.disabled",
              textDecoration: "line-through",
              fontSize: "0.62rem",
              lineHeight: 1.1,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {originalValue}
          </Typography>
          <Typography
            variant="caption"
            sx={{
              color: chipColors.modified,
              fontWeight: 600,
              fontSize: "0.68rem",
              lineHeight: 1.1,
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            {value}
          </Typography>
        </Box>
      ) : (
        <Typography
          variant="caption"
          sx={{
            color: modified ? chipColors.modified : "text.secondary",
            fontWeight: modified ? 600 : 400,
            fontSize: "0.68rem",
            lineHeight: 1.3,
            textAlign: "right",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {value}
        </Typography>
      )}
    </Box>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
      <Box
        sx={{
          width: 12,
          height: 12,
          borderRadius: "50%",
          bgcolor: color,
        }}
      />
      <Typography variant="caption">{label}</Typography>
    </Box>
  );
}

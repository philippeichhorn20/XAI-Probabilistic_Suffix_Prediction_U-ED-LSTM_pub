import { Chip } from "@mui/material";
import { chipColors } from "../theme";

interface Props {
  index: number;
  label: string;
  isPrefix: boolean;
  isModified: boolean;
  onClick: () => void;
}

export default function EventChip({
  index,
  label,
  isPrefix,
  isModified,
  onClick,
}: Props) {
  const bg = isModified
    ? chipColors.modified
    : isPrefix
      ? chipColors.prefix
      : chipColors.suffix;

  const truncated = label.length > 18 ? label.slice(0, 16) + ".." : label;

  return (
    <Chip
      label={`${index + 1}. ${truncated}`}
      onClick={onClick}
      sx={{
        bgcolor: bg,
        color: "#fff",
        fontWeight: 500,
        border: isModified ? "2px dashed #cc6600" : undefined,
        cursor: "pointer",
        "&:hover": {
          opacity: 0.85,
          transform: "scale(1.04)",
        },
        transition: "transform 0.15s, opacity 0.15s",
      }}
    />
  );
}

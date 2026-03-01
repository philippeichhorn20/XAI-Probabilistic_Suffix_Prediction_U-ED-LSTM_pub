import { createTheme } from "@mui/material/styles";

const theme = createTheme({
  palette: {
    primary: { main: "#1f77b4" },
    secondary: { main: "#ff7f0e" },
    background: { default: "#f5f5f5" },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  },
});

export default theme;

export const chipColors = {
  prefix: "#1f77b4",
  suffix: "#aaaaaa",
  modified: "#ff7f0e",
} as const;

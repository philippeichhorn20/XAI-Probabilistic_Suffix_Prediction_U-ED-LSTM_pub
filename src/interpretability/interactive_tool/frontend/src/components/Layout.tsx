import type { ReactNode } from "react";
import { AppBar, Box, Drawer, Toolbar, Typography } from "@mui/material";

const DRAWER_WIDTH = 320;

interface Props {
  sidebar: ReactNode;
  children: ReactNode;
}

export default function Layout({ sidebar, children }: Props) {
  return (
    <Box sx={{ display: "flex" }}>
      <AppBar
        position="fixed"
        sx={{ zIndex: (t) => t.zIndex.drawer + 1 }}
      >
        <Toolbar>
          <Typography variant="h6" noWrap>
            Interactive Case Explorer
          </Typography>
        </Toolbar>
      </AppBar>

      <Drawer
        variant="permanent"
        sx={{
          width: DRAWER_WIDTH,
          flexShrink: 0,
          "& .MuiDrawer-paper": {
            width: DRAWER_WIDTH,
            boxSizing: "border-box",
          },
        }}
      >
        <Toolbar />
        {sidebar}
      </Drawer>

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          mt: "64px",
          minHeight: "calc(100vh - 64px)",
          overflow: "auto",
        }}
      >
        {children}
      </Box>
    </Box>
  );
}

import { create } from "zustand";
import type {
  ModelInfo,
  EditableCase,
  EventData,
  CategoricalFeatureInfo,
  NumericalFeatureInfo,
  PredictResponse,
  AttributionResponse,
} from "../api/types";
import {
  fetchModels,
  fetchCases,
  loadCase as apiLoadCase,
  predict as apiPredict,
  computeAttribution as apiAttribution,
} from "../api/client";

// Deep-clone an EditableCase (events are plain objects)
function cloneCase(c: EditableCase): EditableCase {
  return JSON.parse(JSON.stringify(c));
}

export interface CaseStore {
  // --- data ---
  models: ModelInfo[];
  caseNames: string[];
  casesTotal: number;

  selectedModel: string;
  selectedCase: string;

  originalCase: EditableCase | null;
  modifiedCase: EditableCase | null;

  catInfo: CategoricalFeatureInfo[];
  numInfo: NumericalFeatureInfo[];
  conceptName: string;
  caseLevelCat: string[];
  caseLevelNum: string[];

  prefixLength: number;

  originalPrediction: PredictResponse | null;
  modifiedPrediction: PredictResponse | null;
  originalAttribution: AttributionResponse | null;
  modifiedAttribution: AttributionResponse | null;

  showAttributions: boolean;

  // --- ui ---
  editingEventIdx: number | null;
  loadingCase: boolean;
  loadingPrediction: boolean;
  loadingAttribution: boolean;
  caseSearch: string;

  // --- actions ---
  loadModels: () => Promise<void>;
  setSelectedModel: (model: string) => void;
  searchCases: (search: string) => Promise<void>;
  loadCaseData: (caseName: string) => Promise<void>;
  setPrefixLength: (len: number) => void;
  setShowAttributions: (v: boolean) => void;
  setEditingEventIdx: (idx: number | null) => void;

  updateEvent: (idx: number, event: EventData) => void;
  updateCaseLevelFeature: (
    type: "categorical" | "numerical",
    featureName: string,
    value: number,
  ) => void;
  resetCase: () => void;

  runPrediction: () => Promise<void>;
  runAttribution: () => Promise<void>;
}

export const useCaseStore = create<CaseStore>((set, get) => ({
  // --- initial state ---
  models: [],
  caseNames: [],
  casesTotal: 0,

  selectedModel: "",
  selectedCase: "",

  originalCase: null,
  modifiedCase: null,

  catInfo: [],
  numInfo: [],
  conceptName: "",
  caseLevelCat: [],
  caseLevelNum: [],

  prefixLength: 3,

  originalPrediction: null,
  modifiedPrediction: null,
  originalAttribution: null,
  modifiedAttribution: null,

  showAttributions: false,

  editingEventIdx: null,
  loadingCase: false,
  loadingPrediction: false,
  loadingAttribution: false,
  caseSearch: "",

  // --- actions ---

  loadModels: async () => {
    const res = await fetchModels();
    set({ models: res.models });
    if (res.models.length > 0 && !get().selectedModel) {
      set({ selectedModel: res.models[0].name });
    }
  },

  setSelectedModel: (model) => {
    set({
      selectedModel: model,
      selectedCase: "",
      caseNames: [],
      casesTotal: 0,
      originalCase: null,
      modifiedCase: null,
      originalPrediction: null,
      modifiedPrediction: null,
      originalAttribution: null,
      modifiedAttribution: null,
      caseSearch: "",
    });
  },

  searchCases: async (search) => {
    const model = get().selectedModel;
    if (!model) return;
    set({ caseSearch: search });
    const res = await fetchCases(model, search, 0, 100);
    set({ caseNames: res.cases, casesTotal: res.total });
  },

  loadCaseData: async (caseName) => {
    const model = get().selectedModel;
    if (!model) return;
    set({ loadingCase: true });
    try {
      const res = await apiLoadCase(model, caseName);
      set({
        selectedCase: caseName,
        originalCase: res.case,
        modifiedCase: cloneCase(res.case),
        catInfo: res.cat_info,
        numInfo: res.num_info,
        conceptName: res.concept_name,
        caseLevelCat: res.case_level_cat,
        caseLevelNum: res.case_level_num,
        prefixLength: Math.min(3, res.case.events.length),
        originalPrediction: null,
        modifiedPrediction: null,
        originalAttribution: null,
        modifiedAttribution: null,
        editingEventIdx: null,
      });
    } finally {
      set({ loadingCase: false });
    }
  },

  setPrefixLength: (len) => set({ prefixLength: len }),
  setShowAttributions: (v) => set({ showAttributions: v }),
  setEditingEventIdx: (idx) => set({ editingEventIdx: idx }),

  updateEvent: (idx, event) => {
    const mc = get().modifiedCase;
    if (!mc) return;
    const updated = cloneCase(mc);
    updated.events[idx] = event;
    set({ modifiedCase: updated });
  },

  updateCaseLevelFeature: (type, featureName, value) => {
    const mc = get().modifiedCase;
    if (!mc) return;
    const updated = cloneCase(mc);
    for (const ev of updated.events) {
      if (type === "categorical") {
        ev.categorical[featureName] = value;
      } else {
        ev.numerical[featureName] = value;
      }
    }
    set({ modifiedCase: updated });
  },

  resetCase: () => {
    const oc = get().originalCase;
    if (!oc) return;
    set({
      modifiedCase: cloneCase(oc),
      originalPrediction: null,
      modifiedPrediction: null,
      originalAttribution: null,
      modifiedAttribution: null,
    });
  },

  runPrediction: async () => {
    const { selectedModel, originalCase, modifiedCase, prefixLength } = get();
    if (!selectedModel || !originalCase || !modifiedCase) return;

    set({ loadingPrediction: true });
    try {
      const [origRes, modRes] = await Promise.all([
        apiPredict(selectedModel, { case: originalCase, prefix_length: prefixLength }),
        apiPredict(selectedModel, { case: modifiedCase, prefix_length: prefixLength }),
      ]);
      set({ originalPrediction: origRes, modifiedPrediction: modRes });
    } finally {
      set({ loadingPrediction: false });
    }
  },

  runAttribution: async () => {
    const { selectedModel, originalCase, modifiedCase, prefixLength } = get();
    if (!selectedModel || !originalCase || !modifiedCase) return;

    set({ loadingAttribution: true });
    try {
      const [origRes, modRes] = await Promise.all([
        apiAttribution(selectedModel, { case: originalCase, prefix_length: prefixLength }),
        apiAttribution(selectedModel, { case: modifiedCase, prefix_length: prefixLength }),
      ]);
      set({ originalAttribution: origRes, modifiedAttribution: modRes });
    } finally {
      set({ loadingAttribution: false });
    }
  },
}));

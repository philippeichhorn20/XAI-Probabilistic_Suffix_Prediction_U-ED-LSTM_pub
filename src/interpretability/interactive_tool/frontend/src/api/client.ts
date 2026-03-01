import axios from "axios";
import type {
  ModelsResponse,
  CasesResponse,
  CaseLoadResponse,
  PredictRequest,
  PredictResponse,
  AttributionRequest,
  AttributionResponse,
} from "./types";

const api = axios.create({ baseURL: "/api" });

export async function fetchModels(): Promise<ModelsResponse> {
  const { data } = await api.get<ModelsResponse>("/models");
  return data;
}

export async function fetchCases(
  model: string,
  search = "",
  offset = 0,
  limit = 100,
): Promise<CasesResponse> {
  const { data } = await api.get<CasesResponse>(`/models/${model}/cases`, {
    params: { search, offset, limit },
  });
  return data;
}

export async function loadCase(
  model: string,
  caseName: string,
): Promise<CaseLoadResponse> {
  const { data } = await api.get<CaseLoadResponse>(
    `/models/${model}/cases/${encodeURIComponent(caseName)}`,
  );
  return data;
}

export async function predict(
  model: string,
  body: PredictRequest,
): Promise<PredictResponse> {
  const { data } = await api.post<PredictResponse>(
    `/models/${model}/predict`,
    body,
  );
  return data;
}

export async function computeAttribution(
  model: string,
  body: AttributionRequest,
): Promise<AttributionResponse> {
  const { data } = await api.post<AttributionResponse>(
    `/models/${model}/attribution`,
    body,
  );
  return data;
}

export async function fetchProcessMap(
  model: string,
  caseName: string,
  prefixLength?: number,
): Promise<string> {
  const { data } = await api.get<string>(`/models/${model}/process-map`, {
    params: { case_name: caseName, prefix_length: prefixLength },
    responseType: "text",
  });
  return data;
}

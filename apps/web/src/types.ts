export type RunEvent = {
  type: string;
  timestamp: string;
  run_id: string | null;
  status?: string;
  stage?: string;
  plugin?: string | null;
  mode?: string;
  index?: number | null;
  message?: string;
  reason?: string;
  response?: string;
  final_response?: string;
  ok?: boolean;
  kind?: string;
  [key: string]: unknown;
};

export type RunCreateResponse = {
  run_id: string;
};

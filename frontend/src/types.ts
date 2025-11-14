export type Role = "user" | "bot" | "system";

export interface ChatMessage {
  role: Role;
  content: string;
  type?: "apology" | "rule-answer" | "system";
}

export interface UploadResponse {
  message: string;
  casesIndexed: number;
}



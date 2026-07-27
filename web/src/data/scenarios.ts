export type Scenario = {
  id: string;
  label: string;
  description: string;
  question: string;
  /** Start a fresh session before sending (topic switch / clean demos). */
  newSession?: boolean;
};

export const SCENARIOS: Scenario[] = [
  {
    id: "grounded",
    label: "Grounded",
    description: "HTTPException arguments with citations",
    question: "What arguments does HTTPException take?",
  },
  {
    id: "abstain",
    label: "Abstain",
    description: "Out-of-corpus Redis question",
    question: "How do I configure Redis connection pooling in FastAPI?",
    newSession: true,
  },
  {
    id: "followup",
    label: "Follow-up",
    description: "Attributes after HTTPException context",
    question: "and what are its attributes?",
  },
  {
    id: "topic",
    label: "Topic switch",
    description: "Jump to UploadFile",
    question: "How do I use UploadFile?",
  },
  {
    id: "multi",
    label: "Multi-query",
    description: "Two topics in one ask",
    question: "What args does HTTPException take and how do I use UploadFile?",
    newSession: true,
  },
  {
    id: "cache",
    label: "Cache hit",
    description: "Repeat grounded question",
    question: "What arguments does HTTPException take?",
  },
];

export const STARTER_TOPICS = [
  "How does Depends work in FastAPI?",
  "How do I run background tasks after the response?",
  "What is the signature of UploadFile?",
];

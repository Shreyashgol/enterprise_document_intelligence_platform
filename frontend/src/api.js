// Thin API client for the FastAPI backend.
const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, { method = "GET", body, isForm } = {}) {
  const opts = { method, headers: {} };
  if (body && !isForm) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  } else if (body && isForm) {
    opts.body = body; // FormData sets its own content-type
  }
  const res = await fetch(`${BASE}${path}`, opts);
  if (!res.ok) {
    let detail;
    try {
      detail = (await res.json()).detail;
    } catch {
      detail = res.statusText;
    }
    throw new Error(typeof detail === "string" ? detail : `Request failed (${res.status})`);
  }
  return res.json();
}

export const api = {
  base: BASE,
  health: () => request("/health"),

  // --- auth ---
  signup: (body) => request("/auth/signup", { method: "POST", body }),
  signin: (body) => request("/auth/signin", { method: "POST", body }),
  google: (body) => request("/auth/google", { method: "POST", body }),
  ner: (text) => request("/ner/extract", { method: "POST", body: { text } }),
  relations: (text) =>
    request("/relations/extract", { method: "POST", body: { text } }),
  analyze: (text) => request("/agent/analyze", { method: "POST", body: { text } }),
  search: (query, k = 5) =>
    request("/search", { method: "POST", body: { query, k } }),
  graphQuery: (pattern) =>
    request("/graph/query", { method: "POST", body: pattern }),
  upload: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/documents/upload", { method: "POST", body: form, isForm: true });
  },
};

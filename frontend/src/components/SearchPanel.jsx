import { useState } from "react";
import { api } from "../api";
import Section from "./Section";

export default function SearchPanel() {
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function run() {
    if (!query.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await api.search(query, 5);
      setHits(res.hits);
    } catch (e) {
      setError(e.message);
      setHits(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div className="card p-5">
        <Section title="Semantic search">
          <div className="flex gap-2">
            <input
              className="input"
              placeholder="Search indexed documents…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && run()}
            />
            <button className="btn-primary shrink-0" onClick={run} disabled={loading}>
              {loading ? "Searching…" : "Search"}
            </button>
          </div>
          <p className="mt-2 text-xs text-slate-400">
            Upload documents first — search runs over the embedding index.
          </p>
          {error && (
            <p className="mt-3 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950/50 dark:text-rose-300">{error}</p>
          )}
        </Section>
      </div>

      {hits && (
        <div className="card animate-fade-in p-5">
          <Section title="Results" count={hits.length}>
            {hits.length === 0 ? (
              <p className="text-sm text-slate-400">No matching documents.</p>
            ) : (
              <ul className="space-y-3">
                {hits.map((h, i) => (
                  <li key={i} className="rounded-lg border border-slate-200 p-3 dark:border-slate-800">
                    <div className="mb-1 flex items-center justify-between">
                      <span className="font-mono text-xs text-slate-400">{h.doc_id}</span>
                      <span className="chip bg-brand-50 text-brand-700 dark:bg-brand-500/15 dark:text-brand-500">
                        {h.score.toFixed(3)}
                      </span>
                    </div>
                    <p className="line-clamp-3 text-sm text-slate-700 dark:text-slate-300">{h.text}</p>
                  </li>
                ))}
              </ul>
            )}
          </Section>
        </div>
      )}
    </div>
  );
}

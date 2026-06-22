import { useState } from "react";
import { api } from "../api";
import Section from "./Section";

const RELATIONS = [
  "",
  "works_for",
  "located_in",
  "owns",
  "signed_contract_with",
  "purchased_from",
];

export default function GraphPanel() {
  const [pattern, setPattern] = useState({ source: "", relation: "", target: "" });
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function run() {
    setLoading(true);
    setError("");
    try {
      const clean = Object.fromEntries(
        Object.entries(pattern).filter(([, v]) => v.trim() !== "")
      );
      setData(await api.graphQuery(clean));
    } catch (e) {
      setError(e.message);
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div className="card p-5">
        <Section title="Knowledge graph query">
          <div className="grid gap-3 sm:grid-cols-3">
            <div>
              <label className="label">Source</label>
              <input
                className="input mt-1"
                placeholder="any"
                value={pattern.source}
                onChange={(e) => setPattern({ ...pattern, source: e.target.value })}
              />
            </div>
            <div>
              <label className="label">Relation</label>
              <select
                className="input mt-1"
                value={pattern.relation}
                onChange={(e) => setPattern({ ...pattern, relation: e.target.value })}
              >
                {RELATIONS.map((r) => (
                  <option key={r} value={r}>
                    {r || "any"}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="label">Target</label>
              <input
                className="input mt-1"
                placeholder="any"
                value={pattern.target}
                onChange={(e) => setPattern({ ...pattern, target: e.target.value })}
              />
            </div>
          </div>
          <button className="btn-primary mt-4" onClick={run} disabled={loading}>
            {loading ? "Querying…" : "Query graph"}
          </button>
          {error && (
            <p className="mt-3 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950/50 dark:text-rose-300">{error}</p>
          )}
        </Section>
      </div>

      {data && (
        <div className="grid gap-6 md:grid-cols-3">
          <div className="card animate-fade-in p-5 md:col-span-2">
            <Section title="Triples" count={data.triples.length}>
              {data.triples.length === 0 ? (
                <p className="text-sm text-slate-400">
                  No matching relationships. Upload documents with people and
                  organizations to populate the graph.
                </p>
              ) : (
                <ul className="space-y-2">
                  {data.triples.map((t, i) => (
                    <li
                      key={i}
                      className="flex flex-wrap items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm dark:border-slate-800 dark:bg-slate-800/50"
                    >
                      <span className="font-semibold dark:text-slate-100">{t.source}</span>
                      <span className="rounded bg-brand-50 px-2 py-0.5 font-mono text-xs text-brand-700 dark:bg-brand-500/15 dark:text-brand-500">
                        {t.relation}
                      </span>
                      <span className="font-semibold dark:text-slate-100">{t.target}</span>
                    </li>
                  ))}
                </ul>
              )}
            </Section>
          </div>

          <div className="card animate-fade-in space-y-4 p-5">
            <Section title="Graph stats">
              <dl className="space-y-2 text-sm">
                <Stat label="Entities" value={data.stats.n_entities} />
                <Stat label="Relations" value={data.stats.n_relations} />
              </dl>
            </Section>
            {data.stats.by_label && Object.keys(data.stats.by_label).length > 0 && (
              <Section title="By label">
                <ul className="space-y-1 text-sm">
                  {Object.entries(data.stats.by_label).map(([k, v]) => (
                    <li key={k} className="flex justify-between text-slate-600">
                      <span>{k}</span>
                      <span className="font-semibold">{v}</span>
                    </li>
                  ))}
                </ul>
              </Section>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2 dark:bg-slate-800/50">
      <span className="text-slate-500 dark:text-slate-400">{label}</span>
      <span className="text-lg font-bold text-slate-800 dark:text-slate-100">{value}</span>
    </div>
  );
}

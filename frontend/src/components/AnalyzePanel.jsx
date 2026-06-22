import { useState } from "react";
import { api } from "../api";
import HighlightedText from "./HighlightedText";
import EntityList from "./EntityList";
import RelationList from "./RelationList";
import Section from "./Section";

const SAMPLE =
  "John Smith works at OpenAI, which is based in San Francisco. " +
  "Acme Corp signed a contract with Globex. Pay $2.5M to billing@acme.com " +
  "or call +1 (555) 987-6543 by 2024-01-15.";

export default function AnalyzePanel() {
  const [text, setText] = useState(SAMPLE);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function run() {
    setLoading(true);
    setError("");
    try {
      setResult(await api.analyze(text));
    } catch (e) {
      setError(e.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* Input */}
      <div className="card p-5">
        <Section title="Document text">
          <textarea
            className="input h-64 resize-y font-mono text-[13px] leading-6"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste document text…"
          />
          <div className="mt-3 flex items-center gap-2">
            <button className="btn-primary" onClick={run} disabled={loading || !text.trim()}>
              {loading ? "Analyzing…" : "Run agent workflow"}
            </button>
            <button className="btn-ghost" onClick={() => setText(SAMPLE)}>
              Load sample
            </button>
          </div>
          {error && (
            <p className="mt-3 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950/50 dark:text-rose-300">
              {error}
            </p>
          )}
        </Section>
      </div>

      {/* Output */}
      <div className="card animate-fade-in space-y-6 p-5">
        {!result && (
          <div className="flex h-full min-h-[200px] items-center justify-center text-sm text-slate-400">
            Results will appear here.
          </div>
        )}
        {result && (
          <>
            {result.trace && (
              <div className="flex flex-wrap items-center gap-1.5 text-xs text-slate-500">
                {result.trace.map((t, i) => (
                  <span key={i} className="flex items-center gap-1.5">
                    <span className="rounded bg-slate-100 px-2 py-0.5 font-mono dark:bg-slate-800">{t}</span>
                    {i < result.trace.length - 1 && <span>→</span>}
                  </span>
                ))}
              </div>
            )}

            <Section title="Highlighted entities" count={result.entities.length}>
              <HighlightedText text={text} entities={result.entities} />
            </Section>

            <Section title="Entities" count={result.entities.length}>
              <EntityList entities={result.entities} />
            </Section>

            <Section title="Relationships" count={result.relations.length}>
              <RelationList relations={result.relations} />
            </Section>

            <Section title="Summary">
              <p className="rounded-lg bg-slate-50 p-3 text-sm leading-6 text-slate-700 dark:bg-slate-800/50 dark:text-slate-300">
                {result.summary}
              </p>
            </Section>

            {result.validation && (
              <Section title="Validation">
                <div
                  className={`flex items-center gap-2 text-sm ${
                    result.validation.is_valid ? "text-emerald-600" : "text-amber-600"
                  }`}
                >
                  <span className="text-lg">
                    {result.validation.is_valid ? "✓" : "⚠"}
                  </span>
                  {result.validation.is_valid
                    ? "Output passed all integrity checks."
                    : `${result.validation.issues.length} issue(s) found.`}
                </div>
              </Section>
            )}
          </>
        )}
      </div>
    </div>
  );
}

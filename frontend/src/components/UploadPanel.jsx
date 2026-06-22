import { useState, useRef } from "react";
import { api } from "../api";
import EntityList from "./EntityList";
import RelationList from "./RelationList";
import Section from "./Section";

export default function UploadPanel({ onIndexed }) {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef();

  async function handleFile(file) {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const res = await api.upload(file);
      setResult(res);
      onIndexed?.();
    } catch (e) {
      setError(e.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="card p-5">
        <Section title="Upload a document">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              handleFile(e.dataTransfer.files[0]);
            }}
            onClick={() => inputRef.current?.click()}
            className={`flex cursor-pointer flex-col items-center justify-center gap-2
              rounded-xl border-2 border-dashed p-10 text-center transition-colors
              ${
                dragOver
                  ? "border-brand-500 bg-brand-50 dark:bg-brand-500/10"
                  : "border-slate-300 bg-slate-50 hover:bg-slate-100 dark:border-slate-700 dark:bg-slate-800/50 dark:hover:bg-slate-800"
              }`}
          >
            <svg className="h-10 w-10 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 7.5 7.5 12M12 7.5V21" />
            </svg>
            <p className="text-sm font-medium text-slate-600">
              Drop a file here, or click to browse
            </p>
            <p className="text-xs text-slate-400">PDF, DOCX, TXT, or EML</p>
            <input
              ref={inputRef}
              type="file"
              accept=".pdf,.docx,.txt,.md,.eml"
              className="hidden"
              onChange={(e) => handleFile(e.target.files[0])}
            />
          </div>
          {loading && <p className="mt-3 text-sm text-slate-500">Processing & indexing…</p>}
          {error && (
            <p className="mt-3 rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950/50 dark:text-rose-300">{error}</p>
          )}
          <p className="mt-3 text-xs text-slate-400">
            Uploaded documents are indexed for semantic search and added to the
            knowledge graph.
          </p>
        </Section>
      </div>

      <div className="card animate-fade-in space-y-6 p-5">
        {!result && (
          <div className="flex h-full min-h-[200px] items-center justify-center text-sm text-slate-400">
            Extracted entities & relations will appear here.
          </div>
        )}
        {result && (
          <>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                  {result.metadata.filename}
                </p>
                <p className="text-xs text-slate-400">
                  doc {result.doc_id} · {result.metadata.n_chars} chars
                </p>
              </div>
              <span className="chip bg-emerald-100 text-emerald-700">indexed ✓</span>
            </div>
            <Section title="Entities" count={result.entities.length}>
              <EntityList entities={result.entities} />
            </Section>
            <Section title="Relationships" count={result.relations.length}>
              <RelationList relations={result.relations} />
            </Section>
          </>
        )}
      </div>
    </div>
  );
}

import { useEffect } from "react";

const STEPS = [
  {
    tab: "Analyze",
    title: "Paste text → instant intelligence",
    body: "Drop in an email, contract clause, or any paragraph and click Run agent workflow. You get color-coded entities (people, orgs, money, dates…), the relationships between them, and a plain-English AI summary.",
    icon: (
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.7}
        d="M4 6h16M4 12h10M4 18h7" />
    ),
  },
  {
    tab: "Upload",
    title: "Turn real documents into data",
    body: "Drag-and-drop a PDF, Word doc, text file, or email. It extracts the text, pulls out entities & relationships, and files it into the knowledge base — instantly searchable and connected.",
    icon: (
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.7}
        d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 7.5 7.5 12M12 7.5V21" />
    ),
  },
  {
    tab: "Search",
    title: "Find documents by meaning",
    body: "Search your uploaded documents by meaning, not just keywords — “vendor agreement” can surface a doc that says “signed a contract with a supplier.”",
    icon: (
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.7}
        d="M21 21l-4.3-4.3M11 19a8 8 0 110-16 8 8 0 010 16z" />
    ),
  },
  {
    tab: "Graph",
    title: "Explore the connections",
    body: "Ask pattern questions like “who works_for OpenAI?” The knowledge graph links facts across different documents — even when no single document stated them together.",
    icon: (
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.7}
        d="M7 7a2 2 0 100-4 2 2 0 000 4zM19 9a2 2 0 100-4 2 2 0 000 4zM7 21a2 2 0 100-4 2 2 0 000 4zm0-12v8m1.5-9.5L17 6m-9 11l9-5" />
    ),
  },
];

export default function HelpModal({ open, onClose }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="card max-h-[88vh] w-full max-w-2xl animate-fade-in overflow-y-auto p-6"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
      >
        {/* Header */}
        <div className="mb-5 flex items-start justify-between">
          <div>
            <h2 className="text-lg font-bold text-slate-800 dark:text-slate-100">
              How to use this app
            </h2>
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
              Turn unstructured documents into structured, connected knowledge —
              in four simple tabs.
            </p>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="rounded-lg p-1.5 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-slate-800"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </div>

        {/* Steps */}
        <ol className="space-y-3">
          {STEPS.map((s, i) => (
            <li
              key={s.tab}
              className="flex gap-4 rounded-xl border border-slate-200 p-4 dark:border-slate-800"
            >
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-brand-50 text-brand-600 dark:bg-brand-500/15 dark:text-brand-500">
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  {s.icon}
                </svg>
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-bold uppercase tracking-wide text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                    {i + 1} · {s.tab}
                  </span>
                  <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                    {s.title}
                  </h3>
                </div>
                <p className="mt-1 text-sm leading-6 text-slate-600 dark:text-slate-300">
                  {s.body}
                </p>
              </div>
            </li>
          ))}
        </ol>

        {/* Tip */}
        <p className="mt-4 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-700 dark:bg-amber-500/10 dark:text-amber-400">
          💡 First action after a quiet period may take ~30–50s while the free
          server wakes up — it's instant after that.
        </p>

        <div className="mt-5 flex justify-end">
          <button className="btn-primary" onClick={onClose}>
            Got it
          </button>
        </div>
      </div>
    </div>
  );
}

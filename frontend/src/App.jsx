import { useEffect, useState, useCallback } from "react";
import { api } from "./api";
import { useAuth } from "./context/AuthContext";
import AnalyzePanel from "./components/AnalyzePanel";
import UploadPanel from "./components/UploadPanel";
import SearchPanel from "./components/SearchPanel";
import GraphPanel from "./components/GraphPanel";
import AuthScreen from "./components/AuthScreen";
import ThemeToggle from "./components/ThemeToggle";
import UserMenu from "./components/UserMenu";

const TABS = [
  { id: "analyze", label: "Analyze" },
  { id: "upload", label: "Upload" },
  { id: "search", label: "Search" },
  { id: "graph", label: "Graph" },
];

export default function App() {
  const { user } = useAuth();
  const [tab, setTab] = useState("analyze");
  const [health, setHealth] = useState(null);

  const refresh = useCallback(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    if (user) refresh();
  }, [refresh, user]);

  // Gate the whole app behind authentication.
  if (!user) return <AuthScreen />;

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/80 backdrop-blur dark:border-slate-800 dark:bg-slate-900/80">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-600 text-white">
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h7l5 5v11a2 2 0 01-2 2z" />
              </svg>
            </div>
            <div>
              <h1 className="text-sm font-bold text-slate-800 dark:text-slate-100">
                Enterprise Document Intelligence
              </h1>
              <p className="text-xs text-slate-400">
                NER · Relations · Knowledge Graph · RAG · Agents
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <StatusBadge health={health} />
            <ThemeToggle />
            <UserMenu />
          </div>
        </div>

        {/* Tabs */}
        <nav className="mx-auto flex max-w-6xl gap-1 px-6">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`-mb-px border-b-2 px-4 py-2.5 text-sm font-medium transition-colors ${
                tab === t.id
                  ? "border-brand-600 text-brand-700 dark:text-brand-500"
                  : "border-transparent text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      {/* Body */}
      <main className="mx-auto max-w-6xl px-6 py-8">
        {tab === "analyze" && <AnalyzePanel />}
        {tab === "upload" && <UploadPanel onIndexed={refresh} />}
        {tab === "search" && <SearchPanel />}
        {tab === "graph" && <GraphPanel />}
      </main>

      <footer className="mx-auto max-w-6xl px-6 pb-8 text-center text-xs text-slate-400">
        Built from first principles — custom tokenizer, BiLSTM NER, relation
        extraction, NetworkX graph, pgvector RAG, Groq Llama 3.3.
      </footer>
    </div>
  );
}

function StatusBadge({ health }) {
  const ok = !!health;
  return (
    <div className="hidden items-center gap-2 text-xs sm:flex">
      <span className={`h-2 w-2 rounded-full ${ok ? "bg-emerald-500" : "bg-rose-500"}`} />
      <span className="text-slate-500 dark:text-slate-400">
        {ok ? (
          <>
            {health.indexed_documents} docs · {health.tagger}
          </>
        ) : (
          "API offline"
        )}
      </span>
    </div>
  );
}

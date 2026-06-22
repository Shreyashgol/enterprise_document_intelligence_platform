import { useEffect, useState } from "react";

// Tab metadata with icons
const TABS = [
  {
    id: "overview",
    label: "Overview",
    title: "Overview & Sign-In",
    icon: (
      <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
      </svg>
    ),
  },
  {
    id: "analyze",
    label: "Analyze",
    title: "Analyze Text & Extraction",
    icon: (
      <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M4 6h16M4 12h10M4 18h7" />
      </svg>
    ),
  },
  {
    id: "upload",
    label: "Upload",
    title: "Document Uploading",
    icon: (
      <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
      </svg>
    ),
  },
  {
    id: "search",
    label: "Search",
    title: "Semantic Search",
    icon: (
      <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    ),
  },
  {
    id: "graph",
    label: "Graph",
    title: "Knowledge Graph Connections",
    icon: (
      <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M18 10a6 6 0 00-6-6c-3.314 0-6 2.686-6 6 0 1.012.25 1.968.697 2.802L4 18v2h2l5.198-2.697c.834.448 1.79.697 2.802.697a6 6 0 006-6z" />
      </svg>
    ),
  },
  {
    id: "mental",
    label: "Mental Model",
    title: "Mental Model & Tips",
    icon: (
      <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 01-2 2h0a2 2 0 01-2-2v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
  },
];

export default function HelpModal({ open, onClose }) {
  const [activeTab, setActiveTab] = useState("overview");

  // Interactive simulations state
  const [analyzeState, setAnalyzeState] = useState("idle"); // idle, running, success
  const [analyzeProgress, setAnalyzeProgress] = useState("");
  const [uploadState, setUploadState] = useState("idle"); // idle, uploading, success
  const [uploadProgress, setUploadProgress] = useState(0);
  const [hoveredNode, setHoveredNode] = useState(null);

  // Keyboard shortcut listener
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  // Run mock analysis workflow
  const startAnalyzeSimulation = () => {
    setAnalyzeState("running");
    const steps = [
      "Tokenizing raw input stream...",
      "Executing BiLSTM NER tagger...",
      "Identifying entities (PERSON, ORG, MONEY)...",
      "Mapping syntactic relationships...",
      "Calling Groq Llama 3.3 for synthesis...",
    ];
    let idx = 0;
    setAnalyzeProgress(steps[0]);
    const interval = setInterval(() => {
      idx++;
      if (idx < steps.length) {
        setAnalyzeProgress(steps[idx]);
      } else {
        clearInterval(interval);
        setAnalyzeState("success");
      }
    }, 450);
  };

  // Run mock upload simulation
  const startUploadSimulation = () => {
    setUploadState("uploading");
    setUploadProgress(0);
    const interval = setInterval(() => {
      setUploadProgress((p) => {
        if (p >= 100) {
          clearInterval(interval);
          setUploadState("success");
          return 100;
        }
        return p + 10;
      });
    }, 150);
  };

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 p-4 backdrop-blur-md"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <style>{`
        @keyframes dash {
          to { stroke-dashoffset: -20; }
        }
        .animate-dash {
          stroke-dasharray: 6;
          animation: dash 1s linear infinite;
        }
      `}</style>

      <div
        className="relative flex flex-col max-h-[85vh] w-full max-w-4xl animate-fade-in overflow-hidden rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-100 dark:border-slate-800 px-6 py-4">
          <div>
            <h2 className="text-xl font-bold tracking-tight text-slate-800 dark:text-slate-100">
              Workspace Guide
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Learn how to utilize custom NLP models, semantic storage, and graphs.
            </p>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="rounded-lg p-1.5 text-slate-400 transition-colors hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-600 dark:hover:text-slate-200"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>
        </div>

        {/* Modal Columns */}
        <div className="flex flex-1 flex-col md:flex-row overflow-hidden">
          {/* Sidebar Nav */}
          <div className="w-full md:w-52 shrink-0 border-b md:border-b-0 md:border-r border-slate-100 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-950/20 p-4">
            <nav className="flex flex-row md:flex-col gap-1 overflow-x-auto md:overflow-x-visible pb-2 md:pb-0">
              {TABS.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setActiveTab(t.id)}
                  className={`flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-semibold tracking-wide transition-all whitespace-nowrap md:w-full ${
                    activeTab === t.id
                      ? "bg-brand-600 text-white shadow-sm"
                      : "text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800"
                  }`}
                >
                  {t.icon}
                  {t.label}
                </button>
              ))}
            </nav>
          </div>

          {/* Right Column: Tab Content */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {/* 1. OVERVIEW */}
            {activeTab === "overview" && (
              <div className="space-y-4 animate-fade-in">
                <div className="space-y-2">
                  <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100">
                    Welcome to Enterprise Intelligence
                  </h3>
                  <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-300">
                    This platform helps you process large amounts of unstructured text documents.
                    It runs standard and custom NLP models to extract entities (such as people, organizations, dates, and currency values)
                    and establishes the connections between them in a knowledge graph.
                  </p>
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="rounded-xl border border-slate-100 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/30 p-4">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Signing In</h4>
                    <ul className="text-xs space-y-1.5 text-slate-600 dark:text-slate-400 list-disc pl-4">
                      <li>Register using your name, email, and password.</li>
                      <li>Or select <strong>Continue with Google</strong> to login in one click.</li>
                      <li>Your session stays persistent on your device.</li>
                    </ul>
                  </div>

                  <div className="rounded-xl border border-slate-100 dark:border-slate-800 bg-slate-50/50 dark:bg-slate-900/30 p-4">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">General Controls</h4>
                    <ul className="text-xs space-y-1.5 text-slate-600 dark:text-slate-400 list-disc pl-4">
                      <li>Toggle between 🌙 Dark and ☀️ Light mode using the theme button on the header.</li>
                      <li>Use the user menu (top-right avatar) to check details or sign out.</li>
                      <li>The online dot confirms your connection to the AI parser backend.</li>
                    </ul>
                  </div>
                </div>

                {/* Mock Header Visual */}
                <div className="rounded-xl border border-slate-200 dark:border-slate-800 p-4 bg-slate-50 dark:bg-slate-900/50 space-y-3">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Header Controls Preview</div>
                  <div className="flex items-center justify-between border-b border-slate-200 dark:border-slate-800 pb-3">
                    <div className="flex items-center gap-2">
                      <div className="h-6 w-6 rounded bg-brand-600 flex items-center justify-center text-white text-[10px] font-bold">EDI</div>
                      <div className="text-[11px] font-bold text-slate-800 dark:text-slate-100">Enterprise Intelligence</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-50 dark:bg-emerald-950/30 text-[9px] text-emerald-600 dark:text-emerald-400 font-semibold border border-emerald-200/50 dark:border-emerald-900/50">
                        <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                        API Online
                      </div>
                      <div className="p-1 rounded bg-slate-200 dark:bg-slate-800 text-[10px] cursor-help" title="Theme switch">🌙</div>
                      <div className="h-6 w-6 rounded-full bg-brand-600 flex items-center justify-center text-white text-[9px] font-bold">JD</div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 2. ANALYZE */}
            {activeTab === "analyze" && (
              <div className="space-y-4 animate-fade-in">
                <div className="space-y-2">
                  <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100">
                    Analyze — Instant NLP Pipeline
                  </h3>
                  <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-300">
                    Drop in raw text—like email snippets, clauses, or memos—to execute a local tokenization and entity recognition (NER) process.
                    The system extracts specific entity tags, links relationships, and requests an AI-generated synthesis summary.
                  </p>
                </div>

                {/* Interactive Simulator */}
                <div className="border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden bg-white dark:bg-slate-950">
                  <div className="bg-slate-50 dark:bg-slate-900 px-4 py-2 border-b border-slate-200 dark:border-slate-800 flex justify-between items-center">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">NLP Workflow Simulator</span>
                    {analyzeState === "success" && (
                      <button
                        onClick={() => setAnalyzeState("idle")}
                        className="text-[10px] text-brand-600 dark:text-brand-400 font-bold hover:underline"
                      >
                        Reset Simulator
                      </button>
                    )}
                  </div>

                  <div className="p-4 space-y-3">
                    {analyzeState === "idle" && (
                      <div className="space-y-3">
                        <div className="font-mono text-xs text-slate-500 dark:text-slate-400 bg-slate-50 dark:bg-slate-900 p-3 rounded-lg border border-slate-100 dark:border-slate-800">
                          "John Smith works at OpenAI, based in San Francisco. Acme Corp signed a contract with Globex. Pay $2.5M by 2024-01-15."
                        </div>
                        <button
                          onClick={startAnalyzeSimulation}
                          className="w-full text-center bg-brand-600 text-white rounded-lg py-2 text-xs font-semibold hover:bg-brand-700 transition-colors"
                        >
                          Run Simulator
                        </button>
                      </div>
                    )}

                    {analyzeState === "running" && (
                      <div className="flex flex-col items-center justify-center py-6 space-y-3">
                        <div className="h-5 w-5 border-2 border-brand-600 border-t-transparent rounded-full animate-spin" />
                        <span className="font-mono text-[11px] text-slate-500 dark:text-slate-400 animate-pulse">
                          {analyzeProgress}
                        </span>
                      </div>
                    )}

                    {analyzeState === "success" && (
                      <div className="space-y-4">
                        {/* Highlighted text preview */}
                        <div className="whitespace-pre-wrap break-words leading-7 text-xs text-slate-800 dark:text-slate-200 bg-slate-50 dark:bg-slate-900/50 p-3 rounded-lg border border-slate-100 dark:border-slate-800">
                          <mark className="rounded px-1 bg-rose-100 text-rose-800 dark:bg-rose-950/40 dark:text-rose-300 font-medium">
                            John Smith
                            <sub className="ml-0.5 align-super text-[8px] font-bold">PERSON</sub>
                          </mark>{" "}
                          works at{" "}
                          <mark className="rounded px-1 bg-indigo-100 text-indigo-800 dark:bg-indigo-950/40 dark:text-indigo-300 font-medium">
                            OpenAI
                            <sub className="ml-0.5 align-super text-[8px] font-bold">ORG</sub>
                          </mark>
                          , based in{" "}
                          <mark className="rounded px-1 bg-fuchsia-100 text-fuchsia-800 dark:bg-fuchsia-950/40 dark:text-fuchsia-300 font-medium">
                            San Francisco
                            <sub className="ml-0.5 align-super text-[8px] font-bold">LOCATION</sub>
                          </mark>
                          .{" "}
                          <mark className="rounded px-1 bg-indigo-100 text-indigo-800 dark:bg-indigo-950/40 dark:text-indigo-300 font-medium">
                            Acme Corp
                            <sub className="ml-0.5 align-super text-[8px] font-bold">ORG</sub>
                          </mark>{" "}
                          signed a contract with{" "}
                          <mark className="rounded px-1 bg-indigo-100 text-indigo-800 dark:bg-indigo-950/40 dark:text-indigo-300 font-medium">
                            Globex
                            <sub className="ml-0.5 align-super text-[8px] font-bold">ORG</sub>
                          </mark>
                          . Pay{" "}
                          <mark className="rounded px-1 bg-emerald-100 text-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300 font-medium">
                            $2.5M
                            <sub className="ml-0.5 align-super text-[8px] font-bold">MONEY</sub>
                          </mark>{" "}
                          by{" "}
                          <mark className="rounded px-1 bg-amber-100 text-amber-800 dark:bg-amber-950/40 dark:text-amber-300 font-medium">
                            2024-01-15
                            <sub className="ml-0.5 align-super text-[8px] font-bold">DATE</sub>
                          </mark>
                          .
                        </div>

                        {/* Relations list preview */}
                        <div className="space-y-1.5">
                          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Extracted Relations</span>
                          <div className="flex flex-wrap gap-1.5">
                            <span className="bg-slate-100 dark:bg-slate-800 text-[10px] text-slate-600 dark:text-slate-300 px-2 py-0.5 rounded border border-slate-200/50 dark:border-slate-700/50 font-medium">
                              John Smith → works_for → OpenAI
                            </span>
                            <span className="bg-slate-100 dark:bg-slate-800 text-[10px] text-slate-600 dark:text-slate-300 px-2 py-0.5 rounded border border-slate-200/50 dark:border-slate-700/50 font-medium">
                              OpenAI → located_in → San Francisco
                            </span>
                            <span className="bg-slate-100 dark:bg-slate-800 text-[10px] text-slate-600 dark:text-slate-300 px-2 py-0.5 rounded border border-slate-200/50 dark:border-slate-700/50 font-medium">
                              Acme Corp → contracted_with → Globex
                            </span>
                          </div>
                        </div>

                        {/* Summary preview */}
                        <div className="space-y-1">
                          <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">AI Summary</span>
                          <p className="bg-slate-50 dark:bg-slate-900 text-[11px] leading-relaxed text-slate-600 dark:text-slate-300 p-2.5 rounded border border-slate-200/50 dark:border-slate-800/50">
                            Summary notes employment details for John Smith at OpenAI in San Francisco, alongside a contractual agreement between Acme Corp and Globex involving a $2.5M obligation due January 15, 2024.
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* 3. UPLOAD */}
            {activeTab === "upload" && (
              <div className="space-y-4 animate-fade-in">
                <div className="space-y-2">
                  <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100">
                    Upload — File Integration Pipeline
                  </h3>
                  <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-300">
                    Drag and drop real documents like PDF, Word (docx), Text, or emails.
                    The server automatically reads the files, maps their content, and logs the relationships in the global index.
                  </p>
                </div>

                {/* Interactive Simulator */}
                <div className="border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden bg-white dark:bg-slate-950">
                  <div className="bg-slate-50 dark:bg-slate-900 px-4 py-2 border-b border-slate-200 dark:border-slate-800 flex justify-between items-center">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">File Integration Simulator</span>
                    {uploadState === "success" && (
                      <button
                        onClick={() => setUploadState("idle")}
                        className="text-[10px] text-brand-600 dark:text-brand-400 font-bold hover:underline"
                      >
                        Reset Upload
                      </button>
                    )}
                  </div>

                  <div className="p-6">
                    {uploadState === "idle" && (
                      <div className="border-2 border-dashed border-slate-200 dark:border-slate-800 rounded-xl p-8 flex flex-col items-center justify-center space-y-3 bg-slate-50/50 dark:bg-slate-900/10">
                        <svg className="h-8 w-8 text-slate-400 animate-bounce" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                        </svg>
                        <div className="text-center">
                          <p className="text-xs font-semibold text-slate-700 dark:text-slate-300">Drag & Drop file to simulate</p>
                          <p className="text-[10px] text-slate-400 mt-1">Accepts PDF, DOCX, TXT, EML</p>
                        </div>
                        <button
                          onClick={startUploadSimulation}
                          className="bg-brand-600 text-white text-xs px-4 py-1.5 rounded-lg font-semibold hover:bg-brand-700 transition-colors shadow-sm"
                        >
                          Simulate Drop (agreement.pdf)
                        </button>
                      </div>
                    )}

                    {uploadState === "uploading" && (
                      <div className="space-y-4">
                        <div className="flex justify-between items-center text-xs">
                          <span className="font-semibold text-slate-600 dark:text-slate-300">Uploading agreement.pdf</span>
                          <span className="font-mono text-brand-600 dark:text-brand-400">{uploadProgress}%</span>
                        </div>
                        <div className="w-full bg-slate-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                          <div className="bg-brand-600 h-full transition-all duration-150" style={{ width: `${uploadProgress}%` }} />
                        </div>
                        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400">
                          <span className="h-1.5 w-1.5 rounded-full bg-brand-500 animate-ping" />
                          {uploadProgress < 40 && "Extracting text encoding..."}
                          {uploadProgress >= 40 && uploadProgress < 85 && "Running NLP Tagging on text nodes..."}
                          {uploadProgress >= 85 && "Writing relations to database..."}
                        </div>
                      </div>
                    )}

                    {uploadState === "success" && (
                      <div className="rounded-xl bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-900/50 p-4 space-y-3 animate-fade-in">
                        <div className="flex items-center gap-2.5">
                          <div className="h-6 w-6 rounded-full bg-emerald-500 flex items-center justify-center text-white text-xs font-bold">✓</div>
                          <div>
                            <h4 className="text-xs font-bold text-slate-800 dark:text-slate-100">agreement.pdf indexed successfully!</h4>
                            <p className="text-[10px] text-slate-400">File cataloged and mapped to the active workspace</p>
                          </div>
                          <span className="ml-auto text-[9px] bg-emerald-500 text-white font-bold px-2 py-0.5 rounded">Indexed ✓</span>
                        </div>
                        <div className="border-t border-emerald-100 dark:border-emerald-900/40 pt-2 flex justify-around text-center text-xs">
                          <div>
                            <div className="font-bold text-emerald-700 dark:text-emerald-400">14</div>
                            <div className="text-[9px] text-slate-400 uppercase tracking-wider">Entities</div>
                          </div>
                          <div className="border-l border-emerald-100 dark:border-emerald-900/40" />
                          <div>
                            <div className="font-bold text-emerald-700 dark:text-emerald-400">5</div>
                            <div className="text-[9px] text-slate-400 uppercase tracking-wider">Relations</div>
                          </div>
                          <div className="border-l border-emerald-100 dark:border-emerald-900/40" />
                          <div>
                            <div className="font-bold text-emerald-700 dark:text-emerald-400">95%</div>
                            <div className="text-[9px] text-slate-400 uppercase tracking-wider">Confidence</div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* 4. SEARCH */}
            {activeTab === "search" && (
              <div className="space-y-4 animate-fade-in">
                <div className="space-y-2">
                  <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100">
                    Search — Semantic Context Matching
                  </h3>
                  <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-300">
                    Query files by conceptual meaning rather than rigid keyword parsing.
                    Using high-dimension embeddings, queries like <strong>"vendor agreement"</strong> will correctly find documents that talk about a <em>"signed contract with a supplier"</em> even if the word "vendor" never appears in the text.
                  </p>
                </div>

                {/* Mock Search Demo */}
                <div className="rounded-xl border border-slate-200 dark:border-slate-800 overflow-hidden bg-slate-50 dark:bg-slate-950 p-4 space-y-3">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Semantic Matching Example</div>
                  
                  {/* Search Bar */}
                  <div className="flex gap-2">
                    <div className="flex-1 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg px-3 py-1.5 flex items-center gap-2 text-xs">
                      <svg className="h-3.5 w-3.5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                      </svg>
                      <span className="text-slate-800 dark:text-slate-100 font-medium">vendor agreement</span>
                    </div>
                  </div>

                  {/* Results */}
                  <div className="space-y-2">
                    <div className="bg-white dark:bg-slate-900 p-3 rounded-lg border border-slate-200 dark:border-slate-800 flex flex-col gap-1.5">
                      <div className="flex justify-between items-center">
                        <span className="text-xs font-bold text-slate-700 dark:text-slate-200">supplier_contracts_2025.pdf</span>
                        <span className="text-[9px] bg-brand-50 text-brand-600 dark:bg-brand-950 dark:text-brand-400 px-1.5 py-0.5 rounded font-bold border border-brand-100 dark:border-brand-900">
                          94% Match
                        </span>
                      </div>
                      <p className="text-[11px] leading-relaxed text-slate-600 dark:text-slate-400">
                        "...This binding agreement dictates SLA terms between the core entity and the <strong className="text-brand-600 dark:text-brand-400 underline">supplier</strong> for software delivery..."
                      </p>
                    </div>

                    <div className="bg-white dark:bg-slate-900 p-3 rounded-lg border border-slate-200 dark:border-slate-800 flex flex-col gap-1.5">
                      <div className="flex justify-between items-center">
                        <span className="text-xs font-bold text-slate-700 dark:text-slate-200">outsourcing_guideline.docx</span>
                        <span className="text-[9px] bg-brand-50 text-brand-600 dark:bg-brand-950 dark:text-brand-400 px-1.5 py-0.5 rounded font-bold border border-brand-100 dark:border-brand-900">
                          78% Match
                        </span>
                      </div>
                      <p className="text-[11px] leading-relaxed text-slate-600 dark:text-slate-400">
                        "...policies regarding third-party <strong className="text-brand-600 dark:text-brand-400 underline">service providers</strong> and contract execution metrics..."
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 5. GRAPH */}
            {activeTab === "graph" && (
              <div className="space-y-4 animate-fade-in">
                <div className="space-y-2">
                  <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100">
                    Graph — Knowledge Graph Visualizer
                  </h3>
                  <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-300">
                    Connections are automatically linked across multiple files to form a centralized network.
                    If Doc A says <em>"Mary works for OpenAI"</em> and Doc B says <em>"OpenAI is located in SF"</em>, the knowledge graph bridges them to reveal the connection: <strong>Mary → works_for → OpenAI → located_in → SF</strong>.
                  </p>
                </div>

                {/* Interactive SVG Graph */}
                <div className="border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden bg-white dark:bg-slate-950">
                  <div className="bg-slate-50 dark:bg-slate-900 px-4 py-2 border-b border-slate-200 dark:border-slate-800 flex justify-between items-center">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Interactive Connections Graph (Hover Nodes)</span>
                  </div>

                  <div className="p-4 flex flex-col items-center">
                    <svg viewBox="0 0 400 180" className="w-full h-40 bg-slate-50 dark:bg-slate-900/30 rounded-lg border border-slate-200 dark:border-slate-800">
                      {/* Lines / Edges */}
                      <g stroke="#cbd5e1" strokeWidth="2">
                        <line x1="80" y1="90" x2="200" y2="50" className="animate-dash" stroke="#6366f1" />
                        <line x1="200" y1="50" x2="320" y2="130" stroke="#6366f1" />
                      </g>

                      {/* Edge Label text backgrounds */}
                      <g className="text-[9px] font-bold">
                        {/* works_for label */}
                        <g transform="translate(140, 70) rotate(-18)">
                          <rect x="-26" y="-8" width="52" height="15" rx="3" fill="#ffffff" stroke="#e2e8f0" strokeWidth="1" className="dark:fill-slate-900 dark:stroke-slate-800" />
                          <text textAnchor="middle" y="3" fill="#64748b" className="dark:fill-slate-400">works_for</text>
                        </g>

                        {/* located_in label */}
                        <g transform="translate(260, 90) rotate(33)">
                          <rect x="-26" y="-8" width="52" height="15" rx="3" fill="#ffffff" stroke="#e2e8f0" strokeWidth="1" className="dark:fill-slate-900 dark:stroke-slate-800" />
                          <text textAnchor="middle" y="3" fill="#64748b" className="dark:fill-slate-400">located_in</text>
                        </g>
                      </g>

                      {/* Node Circle 1: John Smith */}
                      <g
                        onMouseEnter={() => setHoveredNode("john")}
                        onMouseLeave={() => setHoveredNode(null)}
                        className="cursor-pointer"
                      >
                        <circle cx="80" cy="90" r="16" fill={hoveredNode === "john" ? "#fda4af" : "#fecdd3"} stroke="#f43f5e" strokeWidth="2" className="transition-all" />
                        <text x="80" y="94" textAnchor="middle" className="text-[9px] font-bold fill-rose-900">JS</text>
                      </g>

                      {/* Node Circle 2: OpenAI */}
                      <g
                        onMouseEnter={() => setHoveredNode("openai")}
                        onMouseLeave={() => setHoveredNode(null)}
                        className="cursor-pointer"
                      >
                        <circle cx="200" cy="50" r="18" fill={hoveredNode === "openai" ? "#a5b4fc" : "#c7d2fe"} stroke="#4f46e5" strokeWidth="2" className="transition-all" />
                        <text x="200" y="54" textAnchor="middle" className="text-[9px] font-bold fill-indigo-900">ORG</text>
                      </g>

                      {/* Node Circle 3: SF */}
                      <g
                        onMouseEnter={() => setHoveredNode("sf")}
                        onMouseLeave={() => setHoveredNode(null)}
                        className="cursor-pointer"
                      >
                        <circle cx="320" cy="130" r="16" fill={hoveredNode === "sf" ? "#f0abfc" : "#f5d0fe"} stroke="#d946ef" strokeWidth="2" className="transition-all" />
                        <text x="320" y="134" textAnchor="middle" className="text-[9px] font-bold fill-fuchsia-900">SF</text>
                      </g>
                    </svg>

                    {/* SVG tooltip description card */}
                    <div className="w-full mt-3 min-h-[48px] bg-slate-50 dark:bg-slate-900/50 p-2.5 rounded-lg border border-slate-200 dark:border-slate-800 text-xs">
                      {hoveredNode === null && (
                        <p className="text-slate-400 text-center italic py-1">Hover over nodes (JS, ORG, SF) to inspect metadata</p>
                      )}
                      {hoveredNode === "john" && (
                        <p className="text-slate-700 dark:text-slate-300">
                          👤 <strong>John Smith (PERSON)</strong>
                          <span className="block text-[10px] text-slate-400 mt-0.5">Found in: bio_profile.pdf · Link: <strong>works_for</strong> OpenAI</span>
                        </p>
                      )}
                      {hoveredNode === "openai" && (
                        <p className="text-slate-700 dark:text-slate-300">
                          🏢 <strong>OpenAI (ORG)</strong>
                          <span className="block text-[10px] text-slate-400 mt-0.5">Found in: bio_profile.pdf, contract.docx · Link: <strong>located_in</strong> San Francisco</span>
                        </p>
                      )}
                      {hoveredNode === "sf" && (
                        <p className="text-slate-700 dark:text-slate-300">
                          📍 <strong>San Francisco (LOCATION)</strong>
                          <span className="block text-[10px] text-slate-400 mt-0.5">Found in: company_registry.txt · Link: <strong>home_of</strong> OpenAI</span>
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 6. MENTAL MODEL & TIPS */}
            {activeTab === "mental" && (
              <div className="space-y-4 animate-fade-in">
                <div className="space-y-2">
                  <h3 className="text-lg font-bold text-slate-800 dark:text-slate-100">
                    Understanding the System Loop
                  </h3>
                  <p className="text-sm leading-relaxed text-slate-600 dark:text-slate-300">
                    A conceptual overview matching user inputs to automation workflows.
                  </p>
                </div>

                {/* Mental Model Table */}
                <div className="overflow-hidden border border-slate-200 dark:border-slate-800 rounded-xl bg-white dark:bg-slate-950">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="bg-slate-50 dark:bg-slate-900 text-slate-400 uppercase tracking-wider border-b border-slate-200 dark:border-slate-800">
                        <th className="p-3 font-bold">You Provide</th>
                        <th className="p-3 font-bold">System Automates</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                      <tr>
                        <td className="p-3 font-semibold text-slate-700 dark:text-slate-300">Upload or paste documents</td>
                        <td className="p-3 text-slate-500 dark:text-slate-400">Extracts text layers, triggers NLP token pipeline.</td>
                      </tr>
                      <tr>
                        <td className="p-3 font-semibold text-slate-700 dark:text-slate-300">Review highlights</td>
                        <td className="p-3 text-slate-500 dark:text-slate-400">Maps entity types (money, names) and relationship models.</td>
                      </tr>
                      <tr>
                        <td className="p-3 font-semibold text-slate-700 dark:text-slate-300">Input questions</td>
                        <td className="p-3 text-slate-500 dark:text-slate-400">Searches semantic indexes and resolves connections dynamically.</td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                {/* Server Warning Tip */}
                <div className="rounded-xl border border-amber-200 dark:border-amber-900/50 bg-amber-50 dark:bg-amber-950/20 p-4 flex gap-3">
                  <div className="h-7 w-7 rounded-lg bg-amber-100 dark:bg-amber-950 flex items-center justify-center text-amber-700 dark:text-amber-400 shrink-0 text-sm font-semibold">💡</div>
                  <div>
                    <h4 className="text-xs font-bold text-amber-800 dark:text-amber-300">Server Cold Start Warning</h4>
                    <p className="text-[11px] leading-relaxed text-amber-700 dark:text-amber-400/90 mt-1">
                      On the free hosting tier, the server falls asleep during inactive periods. The initial action after a quiet period will take 30 to 50 seconds to complete while backend engines initialize. Subsequent actions will be immediate.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-slate-100 dark:border-slate-800 px-6 py-4 flex items-center justify-between bg-slate-50/50 dark:bg-slate-950/20">
          <p className="text-[10px] text-slate-400">
            Powered by local BiLSTM Tagging & pgvector RAG
          </p>
          <button
            className="btn-primary py-1.5 px-4 text-xs font-semibold"
            onClick={onClose}
          >
            Got it, thanks
          </button>
        </div>
      </div>
    </div>
  );
}

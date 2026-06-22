import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import ThemeToggle from "./ThemeToggle";
import GoogleButton from "./GoogleButton";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export default function AuthScreen() {
  const { signin, signup } = useAuth();
  const [mode, setMode] = useState("signin"); // "signin" | "signup"
  const [form, setForm] = useState({ name: "", email: "", password: "", confirm: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const isSignup = mode === "signup";
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });

  function validate() {
    if (isSignup && !form.name.trim()) return "Please enter your name.";
    if (!EMAIL_RE.test(form.email)) return "Please enter a valid email address.";
    if (form.password.length < 6) return "Password must be at least 6 characters.";
    if (isSignup && form.password !== form.confirm) return "Passwords do not match.";
    return "";
  }

  async function submit(e) {
    e.preventDefault();
    const v = validate();
    if (v) {
      setError(v);
      return;
    }
    setError("");
    setLoading(true);
    try {
      if (isSignup) {
        await signup(form);
      } else {
        await signin(form);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function switchMode() {
    setMode(isSignup ? "signin" : "signup");
    setError("");
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50 p-4 dark:bg-slate-950">
      <div className="absolute right-4 top-4">
        <ThemeToggle />
      </div>

      <div className="w-full max-w-md animate-fade-in">
        {/* Brand */}
        <div className="mb-6 flex flex-col items-center gap-3 text-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-brand-600 text-white">
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h7l5 5v11a2 2 0 01-2 2z" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-bold text-slate-800 dark:text-slate-100">
              Enterprise Document Intelligence
            </h1>
            <p className="text-sm text-slate-400">
              {isSignup ? "Create your account" : "Sign in to your workspace"}
            </p>
          </div>
        </div>

        {/* Card */}
        <div className="card p-6">
          {/* Tabs */}
          <div className="mb-5 grid grid-cols-2 rounded-lg bg-slate-100 p-1 dark:bg-slate-800">
            {["signin", "signup"].map((m) => (
              <button
                key={m}
                onClick={() => {
                  setMode(m);
                  setError("");
                }}
                className={`rounded-md py-1.5 text-sm font-medium transition-colors ${
                  mode === m
                    ? "bg-white text-brand-700 shadow-sm dark:bg-slate-900 dark:text-brand-500"
                    : "text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
                }`}
              >
                {m === "signin" ? "Sign in" : "Sign up"}
              </button>
            ))}
          </div>

          <form onSubmit={submit} className="space-y-4">
            {isSignup && (
              <Field label="Full name">
                <input className="input" value={form.name} onChange={set("name")}
                  placeholder="Jane Doe" autoComplete="name" />
              </Field>
            )}
            <Field label="Email">
              <input className="input" type="email" value={form.email} onChange={set("email")}
                placeholder="you@company.com" autoComplete="email" />
            </Field>
            <Field label="Password">
              <input className="input" type="password" value={form.password} onChange={set("password")}
                placeholder="••••••••"
                autoComplete={isSignup ? "new-password" : "current-password"} />
            </Field>
            {isSignup && (
              <Field label="Confirm password">
                <input className="input" type="password" value={form.confirm} onChange={set("confirm")}
                  placeholder="••••••••" autoComplete="new-password" />
              </Field>
            )}

            {error && (
              <p className="rounded-lg bg-rose-50 px-3 py-2 text-sm text-rose-700 dark:bg-rose-950/50 dark:text-rose-300">
                {error}
              </p>
            )}

            <button className="btn-primary w-full" disabled={loading}>
              {loading ? "Please wait…" : isSignup ? "Create account" : "Sign in"}
            </button>
          </form>

          {/* Divider */}
          <div className="my-5 flex items-center gap-3">
            <span className="h-px flex-1 bg-slate-200 dark:bg-slate-800" />
            <span className="text-xs font-medium text-slate-400">OR</span>
            <span className="h-px flex-1 bg-slate-200 dark:bg-slate-800" />
          </div>

          {/* Google OAuth */}
          <GoogleButton onError={setError} />

          <p className="mt-5 text-center text-sm text-slate-500">
            {isSignup ? "Already have an account?" : "Don't have an account?"}{" "}
            <button onClick={switchMode} className="font-semibold text-brand-600 hover:underline">
              {isSignup ? "Sign in" : "Sign up"}
            </button>
          </p>
        </div>

        <p className="mt-4 text-center text-xs text-slate-400">
          Accounts are stored locally in your browser for this demo.
        </p>
      </div>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <div>
      <label className="label mb-1 block">{label}</label>
      {children}
    </div>
  );
}

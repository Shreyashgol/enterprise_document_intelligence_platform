import { useEffect, useRef, useState } from "react";
import { useAuth } from "../context/AuthContext";

function initials(name = "", email = "") {
  const base = name.trim() || email;
  const parts = base.split(/\s+/).filter(Boolean);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return base.slice(0, 2).toUpperCase();
}

export default function UserMenu() {
  const { user, signout } = useAuth();
  const [open, setOpen] = useState(false);
  const ref = useRef();

  useEffect(() => {
    function onClick(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  if (!user) return null;

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex h-9 w-9 items-center justify-center overflow-hidden rounded-full
                   bg-brand-600 text-xs font-bold text-white transition-transform hover:scale-105"
        title={user.email}
      >
        {user.picture ? (
          <img src={user.picture} alt="" className="h-full w-full object-cover"
            referrerPolicy="no-referrer" />
        ) : (
          initials(user.name, user.email)
        )}
      </button>

      {open && (
        <div
          className="absolute right-0 mt-2 w-56 animate-fade-in overflow-hidden rounded-xl
                     border border-slate-200 bg-white shadow-lg
                     dark:border-slate-700 dark:bg-slate-900"
        >
          <div className="border-b border-slate-100 px-4 py-3 dark:border-slate-800">
            <p className="truncate text-sm font-semibold text-slate-800 dark:text-slate-100">
              {user.name || "Account"}
            </p>
            <p className="truncate text-xs text-slate-400">{user.email}</p>
          </div>
          <button
            onClick={() => {
              setOpen(false);
              signout();
            }}
            className="flex w-full items-center gap-2 px-4 py-2.5 text-left text-sm
                       text-rose-600 transition-colors hover:bg-slate-50 dark:hover:bg-slate-800"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8}
                d="M15 12H3m0 0l4-4m-4 4l4 4m6-11h4a2 2 0 012 2v10a2 2 0 01-2 2h-4" />
            </svg>
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}

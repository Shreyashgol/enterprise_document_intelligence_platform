import { useEffect, useRef, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { useTheme } from "../context/ThemeContext";

const CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;
const GSI_SRC = "https://accounts.google.com/gsi/client";

// Decode a Google ID-token (JWT) payload, UTF-8 safe.
function decodeJwt(token) {
  const b64 = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
  const json = decodeURIComponent(
    atob(b64)
      .split("")
      .map((c) => "%" + c.charCodeAt(0).toString(16).padStart(2, "0"))
      .join("")
  );
  return JSON.parse(json);
}

function loadGsi() {
  return new Promise((resolve, reject) => {
    if (window.google?.accounts?.id) return resolve();
    const existing = document.querySelector(`script[src="${GSI_SRC}"]`);
    if (existing) {
      existing.addEventListener("load", () => resolve());
      existing.addEventListener("error", () => reject(new Error("load error")));
      return;
    }
    const s = document.createElement("script");
    s.src = GSI_SRC;
    s.async = true;
    s.defer = true;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("Failed to load Google sign-in."));
    document.head.appendChild(s);
  });
}

export default function GoogleButton({ onError }) {
  const { signinWithGoogle } = useAuth();
  const { theme } = useTheme();
  const ref = useRef(null);
  const [ready, setReady] = useState(false);

  // Real Google Identity Services flow (requires VITE_GOOGLE_CLIENT_ID).
  useEffect(() => {
    if (!CLIENT_ID || !ref.current) return;
    let cancelled = false;

    loadGsi()
      .then(() => {
        if (cancelled || !ref.current) return;
        window.google.accounts.id.initialize({
          client_id: CLIENT_ID,
          callback: async (resp) => {
            try {
              const p = decodeJwt(resp.credential);
              await signinWithGoogle({
                name: p.name,
                email: p.email,
                picture: p.picture,
              });
            } catch (e) {
              onError?.(e.message || "Google sign-in failed.");
            }
          },
        });
        ref.current.innerHTML = "";
        window.google.accounts.id.renderButton(ref.current, {
          type: "standard",
          theme: theme === "dark" ? "filled_black" : "outline",
          size: "large",
          width: 320,
          text: "continue_with",
          shape: "pill",
        });
        setReady(true);
      })
      .catch((e) => onError?.(e.message));

    return () => {
      cancelled = true;
    };
  }, [theme, signinWithGoogle, onError]);

  // Configured → render the official Google button.
  if (CLIENT_ID) {
    return (
      <div className="flex justify-center">
        <div ref={ref} />
        {!ready && (
          <div className="btn-ghost w-full justify-center">Loading Google…</div>
        )}
      </div>
    );
  }

  // Not configured → a styled fallback so the UI is functional out of the box.
  return (
    <button
      type="button"
      onClick={() =>
        signinWithGoogle({
          name: "Demo Google User",
          email: "demo.user@gmail.com",
          picture: null,
        }).catch((e) => onError?.(e.message))
      }
      className="btn-ghost w-full"
      title="Set VITE_GOOGLE_CLIENT_ID to enable real Google sign-in"
    >
      <GoogleIcon />
      Continue with Google
      <span className="ml-1 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold text-slate-500 dark:bg-slate-700 dark:text-slate-300">
        demo
      </span>
    </button>
  );
}

function GoogleIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 24 24">
      <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 01-2.2 3.32v2.77h3.57c2.08-1.92 3.27-4.74 3.27-8.1z" />
      <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84A11 11 0 0012 23z" />
      <path fill="#FBBC05" d="M5.84 14.1a6.6 6.6 0 010-4.2V7.06H2.18a11 11 0 000 9.88l3.66-2.84z" />
      <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1A11 11 0 002.18 7.06l3.66 2.84C6.71 7.31 9.14 5.38 12 5.38z" />
    </svg>
  );
}

import { createContext, useContext, useState } from "react";
import { api } from "../api";

// Authentication backed by the FastAPI `/auth/*` endpoints — users are
// persisted server-side in the `users` table (PostgreSQL). The session token
// returned by the API is kept in localStorage so the user stays signed in
// across reloads.

const AuthContext = createContext(null);
const SESSION_KEY = "edi_session";

export function AuthProvider({ children }) {
  const [session, setSession] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(SESSION_KEY));
    } catch {
      return null;
    }
  });

  function persist(res) {
    // res = { token, user }
    localStorage.setItem(SESSION_KEY, JSON.stringify(res));
    setSession(res);
  }

  async function signup({ name, email, password }) {
    persist(await api.signup({ name, email, password }));
  }

  async function signin({ email, password }) {
    persist(await api.signin({ email, password }));
  }

  async function signinWithGoogle(profile) {
    persist(
      await api.google({
        email: profile.email,
        name: profile.name,
        picture: profile.picture,
      })
    );
  }

  function signout() {
    localStorage.removeItem(SESSION_KEY);
    setSession(null);
  }

  return (
    <AuthContext.Provider
      value={{
        user: session?.user || null,
        token: session?.token || null,
        signup,
        signin,
        signinWithGoogle,
        signout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}

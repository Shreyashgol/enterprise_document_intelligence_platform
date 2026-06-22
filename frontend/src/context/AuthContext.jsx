import { createContext, useContext, useState } from "react";

// Client-side authentication (localStorage).
// Passwords are SHA-256 hashed before storage — never kept in plaintext.
// This is a self-contained demo auth layer; swap `signup`/`signin` for real
// API calls (e.g. POST /auth/signup) to back it with a server + JWT later.

const AuthContext = createContext(null);
const USERS_KEY = "edi_users";
const SESSION_KEY = "edi_session";

async function sha256(text) {
  const data = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(digest)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function loadUsers() {
  try {
    return JSON.parse(localStorage.getItem(USERS_KEY)) || {};
  } catch {
    return {};
  }
}

function saveUsers(users) {
  localStorage.setItem(USERS_KEY, JSON.stringify(users));
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(SESSION_KEY));
    } catch {
      return null;
    }
  });

  function startSession(session) {
    localStorage.setItem(SESSION_KEY, JSON.stringify(session));
    setUser(session);
  }

  async function signup({ name, email, password }) {
    const key = email.trim().toLowerCase();
    const users = loadUsers();
    if (users[key]) {
      throw new Error("An account with this email already exists.");
    }
    users[key] = {
      name: name.trim(),
      email: key,
      passwordHash: await sha256(password),
      createdAt: Date.now(),
    };
    saveUsers(users);
    startSession({ name: name.trim(), email: key });
  }

  async function signin({ email, password }) {
    const key = email.trim().toLowerCase();
    const record = loadUsers()[key];
    const hash = await sha256(password);
    if (!record || record.passwordHash !== hash) {
      throw new Error("Invalid email or password.");
    }
    startSession({ name: record.name, email: key });
  }

  // OAuth (Google): no password — the identity is asserted by the provider.
  async function signinWithGoogle(profile) {
    const key = (profile.email || "").trim().toLowerCase();
    if (!key) throw new Error("Google account did not return an email.");
    const users = loadUsers();
    if (!users[key]) {
      users[key] = {
        name: profile.name || key,
        email: key,
        provider: "google",
        picture: profile.picture || null,
        createdAt: Date.now(),
      };
      saveUsers(users);
    }
    startSession({
      name: users[key].name,
      email: key,
      picture: users[key].picture || profile.picture || null,
      provider: "google",
    });
  }

  function signout() {
    localStorage.removeItem(SESSION_KEY);
    setUser(null);
  }

  return (
    <AuthContext.Provider
      value={{ user, signup, signin, signinWithGoogle, signout }}
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

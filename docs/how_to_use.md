The scenario

Imagine you're an analyst, paralegal, or ops person drowning in contracts, invoices, and emails. You need to pull out who/what/when/how much and connect the dots across documents. That's what this app does.

---
1. Sign in

You open the Vercel URL and hit a login screen. You can:
- Sign up with name + email + password, or
- Continue with Google (one click).

Your account is saved, so next time it remembers you. There's a 🌙/☀️ toggle (top-right) to switch dark/light, and your avatar menu to sign out.

Once in, you land on a workspace with four tabs: Analyze · Upload · Search · Graph. A little status dot shows the system is online.

---
2. Analyze — paste text, get instant intelligence

This is the "wow" tab. You paste any text (an email, a contract clause, a paragraph) and click Run agent workflow. In a moment you get:

- Highlighted entities — people, companies, locations, products, emails, phones, dates, and money are color-coded right in your text.
- Entity list — a clean tally (e.g., John Smith · PERSON, $2.5M · MONEY).
- Relationships — the connections it found, like:
John Smith → works_for → OpenAI · OpenAI → located_in → San Francisco
- Summary — a plain-English recap written by an AI (Groq Llama 3.3).
- Validation — a ✓ confirming the extraction is consistent.

▎ Real example: paste "John Smith works at OpenAI in San Francisco. Acme signed a contract with Globex. Pay $2.5M to billing@acme.com by 2024-01-15." → it highlights every entity, maps the relationships, and summarizes the deal.

Who uses it: anyone who wants to quickly understand "what's in this text" without reading it closely.

---
3. Upload — turn real documents into data

Drag-and-drop a PDF, Word doc, text file, or email (or click to browse). The app:
- Extracts the text automatically (you don't convert anything),
- Pulls out all the entities and relationships,
- And files it into the system's memory — so it's now searchable and connected to everything else you've uploaded.

You'll see the extracted entities/relations and an "indexed ✓" confirmation.

Who uses it: someone processing a stack of real files — contracts, resumes, invoices.

---
4. Search — find documents by meaning

Type a question or phrase like "cloud services contract" and it returns the most relevant uploaded documents — ranked by meaning, not just keyword matching. So searching "agreement with a vendor" can surface a doc that says "signed a contract with a supplier."

Who uses it: "I know we have a document about X somewhere" — find it fast.

---
5. Graph — explore the connections

Everything you upload builds a knowledge graph — a web of who's connected to whom. In this tab you ask pattern questions:
- "Show me everyone who works_for OpenAI"
- "What is OpenAI located_in?"
- Or leave fields blank to see all relationships.

You get the matching connections plus stats (how many entities, by type). The power: it connects facts across different documents — Document A says "Mary works at OpenAI," Document B says "OpenAI is in SF," and the graph links them even though no single document said both.

Who uses it: investigators, analysts, anyone asking "how is X connected to Y?"

---
The mental model

Think of it as a smart filing cabinet that reads for you:

┌──────────────────────────────────┬──────────────────────────────────────────────────────────┐
│              You do              │                       The app does                       │
├──────────────────────────────────┼──────────────────────────────────────────────────────────┤
│ Paste or upload a document       │ Reads it, extracts the key facts                         │
├──────────────────────────────────┼──────────────────────────────────────────────────────────┤
│                                  │ Tags people, orgs, money, dates, etc.                    │
├──────────────────────────────────┼──────────────────────────────────────────────────────────┤
│                                  │ Maps how they relate                                     │
├──────────────────────────────────┼──────────────────────────────────────────────────────────┤
│                                  │ Writes a summary                                         │
├──────────────────────────────────┼──────────────────────────────────────────────────────────┤
│                                  │ Remembers it for search + connects it to everything else │
├──────────────────────────────────┼──────────────────────────────────────────────────────────┤
│ Ask a question (search or graph) │ Finds the answer across all your documents               │
└──────────────────────────────────┴──────────────────────────────────────────────────────────┘

So a 20-page contract becomes: a list of parties, amounts, dates, the relationships between them, a one-paragraph summary, and a searchable, connected entry in your knowledge base — in seconds instead of an afternoon.

---
One practical note for your live demo: on the free hosting tier the server "sleeps" when idle, so the very first action after a quiet period takes ~30–50 seconds to wake up — after that it's instant.
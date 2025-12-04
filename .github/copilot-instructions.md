<!-- .github/copilot-instructions.md - tailored instructions for AI coding agents -->
# Copilot / AI Agent Instructions — nmap-ai-interface

This repository is a small Vite + React frontend that simulates an "NMAP-AI" command generator (no backend included). The file to focus on for feature work is `src/App.jsx`, which contains the UI, state model and the simulated AI response logic.

Key facts (big picture)
- **Type:** Frontend-only React app (Vite). No server API present in this repo.
- **Entry:** `src/main.jsx` mounts `<App />`.
- **Core UI + logic:** `src/App.jsx` — handles user input, message state, simulated assistant responses, and clipboard interaction.
- **Styling:** `src/index.css` + `src/App.css`. Tailwind directives are present but commented out. The project includes `tailwindcss` in `package.json` but Tailwind is not wired in by default.

Developer workflows (exact commands)
- Install dependencies: `npm install` (or `pnpm`/`yarn` if preferred).
- Start dev server (HMR): `npm run dev` (Vite default port, typically `http://localhost:5173`).
- Build production: `npm run build`.
- Preview build locally: `npm run preview`.
- Lint: `npm run lint` (uses `eslint.config.js`).

Project-specific patterns & conventions
- Message model: components expect message objects shaped like:

```js
{
  id: Number,
  type: 'assistant' | 'user',
  content: String,
  timestamp: String,
  // optional when assistant returns a command
  command?: String,
  complexity?: 'easy'|'medium'|'hard'
}
```

- UI relies on these fields. When integrating a real AI backend, ensure API responses map into this shape (attach `command` and `complexity` when available).
- Simulated AI behavior lives inside `handleSubmit()` in `src/App.jsx` — it currently uses `setTimeout()` + randomized example commands. Replace this block to call your real RAG/LLM/MCP backend.
- Clipboard: copying commands uses `navigator.clipboard.writeText(cmd)` — keep this for copy behavior.
- Key handling: Enter to send (unless Shift+Enter); see `handleKeyPress`.

Integration points & where to change code
- To call a backend: edit `handleSubmit` in `src/App.jsx`. Replace the `setTimeout()` stub with a `fetch()` or SDK call. After receiving the response, push an assistant message object (matching the message model above) into `messages` via `setMessages`.
- If adding Tailwind: uncomment Tailwind directives inside `src/index.css` and add PostCSS/Tailwind build steps; currently Tailwind is listed in `devDependencies` but not configured.
- Icons come from `lucide-react` (imported in `src/App.jsx`). Keep imports consistent with tree-shaking patterns (named imports are used).

ESLint and coding style
- ESLint uses `eslint.config.js` with browser globals and React/JXE rules. Pay attention to the custom `no-unused-vars` rule which ignores names starting with uppercase or underscore.

Files to inspect first when making changes
- `src/App.jsx` — main area for logic, messages, and UI.
- `src/main.jsx` — app bootstrap.
- `src/index.css` — global styles; contains commented Tailwind directives.
- `package.json` — scripts and deps (`dev`, `build`, `preview`, `lint`).

Short examples (how to return a command from backend):

```js
// after getting a response from your API
const assistantMessage = {
  id: Date.now(),
  type: 'assistant',
  content: apiResp.summaryText,
  command: apiResp.commandString, // shown in UI
  complexity: apiResp.complexity || 'medium',
  timestamp: new Date().toLocaleTimeString()
};
setMessages(prev => [...prev, assistantMessage]);
```

Notes / gotchas discovered
- The UI text is in French and English; preserve locale when adjusting copy.
- The footer references "Knowledge Graph RAG • Fine-Tuning • MCP Protocol" but no RAG/MCP integration exists in this repo — treat this as a product hint, not implemented code.
- Tailwind is not active by default despite many Tailwind classes in JSX; if you enable Tailwind, visual output will change significantly.

If anything below is unclear or you'd like instructions expanded (API contract, message examples, or a minimal backend stub), tell me which area to expand and I will iterate.

— End of copilot instructions for `nmap-ai-interface`.

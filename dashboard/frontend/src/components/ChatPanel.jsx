import { useState } from "react";
import { api } from "../api.js";

// Defaults to Haiku rather than agent.copilot's own Sonnet default -- this is
// the panel someone will click repeatedly while poking at the demo, and each
// click is a real API call; Sonnet is one dropdown selection away if quality
// matters more than cost for a given question.
const MODELS = [
  { id: "claude-haiku-4-5-20251001", label: "Haiku (fast, cheap)" },
  { id: "claude-sonnet-5", label: "Sonnet (higher quality)" },
];

export default function ChatPanel() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [model, setModel] = useState(MODELS[0].id);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState(null);

  const send = async () => {
    const question = input.trim();
    if (!question || sending) return;

    const nextMessages = [...messages, { role: "user", content: question }];
    setMessages(nextMessages);
    setInput("");
    setSending(true);
    setError(null);

    try {
      const history = messages.map((m) => ({ role: m.role, content: m.content }));
      const result = await api.chat(question, history.length ? history : null, model);
      setMessages([...nextMessages, { role: "assistant", content: result.answer, trace: result.trace }]);
    } catch (e) {
      setError(e.message);
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="panel chat-panel">
      <div className="controls">
        <label>
          Model:{" "}
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <p className="muted">
            Ask about a SKU's forecast, recent sales, inventory, or anomalies -- e.g. "Is FOODS_1_018 at
            risk of stocking out?"
          </p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`chat-message chat-${m.role}`}>
            <div className="chat-bubble">{m.content}</div>
            {m.trace && m.trace.length > 0 && (
              <details className="chat-trace">
                <summary>{m.trace.length} tool call(s)</summary>
                {m.trace.map((t, j) => (
                  <div key={j} className="trace-call">
                    <code>
                      {t.tool}({JSON.stringify(t.input)})
                    </code>
                    {t.is_error && <span className="trace-error"> error</span>}
                  </div>
                ))}
              </details>
            )}
          </div>
        ))}
        {sending && <p className="muted">Thinking...</p>}
      </div>

      {error && <p className="error">{error}</p>}

      <div className="chat-input">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Ask the copilot..."
        />
        <button onClick={send} disabled={sending}>
          Send
        </button>
      </div>
    </div>
  );
}

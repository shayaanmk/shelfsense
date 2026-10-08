import { useState } from "react";
import SkuExplorer from "./components/SkuExplorer.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import CommercePanel from "./components/CommercePanel.jsx";

const TABS = [
  { id: "explorer", label: "Forecast & Anomalies" },
  { id: "chat", label: "Copilot Chat" },
  { id: "commerce", label: "Commerce Simulation" },
];

export default function App() {
  const [tab, setTab] = useState("explorer");

  return (
    <div className="app">
      <header className="app-header">
        <h1>shelfsense</h1>
        <p className="subtitle">Supply chain forecasting + agent copilot</p>
      </header>

      <nav className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`tab ${tab === t.id ? "active" : ""}`}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className="app-main">
        {tab === "explorer" && <SkuExplorer />}
        {tab === "chat" && <ChatPanel />}
        {tab === "commerce" && <CommercePanel />}
      </main>
    </div>
  );
}

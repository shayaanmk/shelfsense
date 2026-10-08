const TYPE_LABEL = {
  demand_spike: "Demand spike",
  demand_drop: "Demand drop",
  stockout: "Stockout",
};

export default function AnomalyList({ anomalies }) {
  const list = anomalies?.anomalies ?? [];

  if (list.length === 0) {
    return <p className="muted">No anomalies flagged in this window.</p>;
  }

  return (
    <ul className="anomaly-list">
      {list.map((a, i) => (
        <li key={i} className={`anomaly anomaly-${a.type}`}>
          <div className="anomaly-head">
            <span className="anomaly-type">{TYPE_LABEL[a.type] ?? a.type}</span>
            <span className="anomaly-date">{a.date}</span>
            {a.severity != null && <span className="anomaly-severity">z={a.severity}</span>}
          </div>
          <div className="anomaly-evidence">
            {Object.entries(a.evidence).map(([k, v]) => (
              <span key={k} className="evidence-chip">
                {k}: {String(v)}
              </span>
            ))}
          </div>
        </li>
      ))}
    </ul>
  );
}

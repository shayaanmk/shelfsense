import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api.js";

export default function CommercePanel() {
  const [state, setState] = useState(null);
  const [log, setLog] = useState([]);
  const [orders, setOrders] = useState([]);
  const [decisions, setDecisions] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const refresh = () =>
    Promise.all([api.commerceState(), api.commerceLog(), api.commerceOrders(), api.commerceDecisions()])
      .then(([s, l, o, d]) => {
        setState(s);
        setLog(l.log);
        setOrders(o.orders);
        setDecisions(d.decisions.slice().reverse());
      })
      .catch((e) => setError(e.message));

  useEffect(() => {
    refresh();
  }, []);

  const advance = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.commerceAdvance();
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const reset = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.commerceReset();
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  if (!state) return <p className="muted">Loading...</p>;

  const belowReorder = state.on_hand <= state.reorder_point;

  return (
    <div className="panel">
      <section className="card">
        <h3>
          {state.sku} &middot; {state.current_date}
          {state.finished && <span className="badge badge-done">simulation complete</span>}
        </h3>
        <div className="stat-row">
          <div className={`stat ${belowReorder ? "stat-warn" : ""}`}>
            <div className="stat-label">On hand</div>
            <div className="stat-value">{state.on_hand.toFixed(1)}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Reorder point</div>
            <div className="stat-value">{state.reorder_point.toFixed(1)}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Target level</div>
            <div className="stat-value">{state.target_level.toFixed(1)}</div>
          </div>
        </div>
        <div className="controls">
          <button onClick={advance} disabled={busy || state.finished}>
            Advance 1 day
          </button>
          <button onClick={reset} disabled={busy} className="secondary">
            Reset simulation
          </button>
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      <section className="card">
        <h3>On-hand inventory over time</h3>
        {log.length === 0 ? (
          <p className="muted">No days simulated yet -- click "Advance 1 day".</p>
        ) : (
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={log} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Line type="monotone" dataKey="on_hand_end_of_day" name="On hand" stroke="#2563eb" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </section>

      <section className="card">
        <h3>Purchase orders ({orders.length})</h3>
        {orders.length === 0 ? (
          <p className="muted">No orders placed yet.</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>PO</th>
                <th>Supplier</th>
                <th>Qty</th>
                <th>Unit $</th>
                <th>Total</th>
                <th>Ordered</th>
                <th>Arrives</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {orders
                .slice()
                .reverse()
                .map((o) => (
                  <tr key={o.po_id}>
                    <td>{o.po_id}</td>
                    <td>{o.supplier_id}</td>
                    <td>{o.quantity}</td>
                    <td>{o.unit_price}</td>
                    <td>{o.total_cost}</td>
                    <td>{o.order_date}</td>
                    <td>{o.expected_arrival_date}</td>
                    <td className={`status status-${o.status}`}>{o.status}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="card">
        <h3>Agent decision log</h3>
        {decisions.length === 0 ? (
          <p className="muted">No decisions logged yet.</p>
        ) : (
          <ul className="decision-list">
            {decisions.map((d, i) => (
              <li key={i} className={`decision decision-${d.action}`}>
                <div className="decision-head">
                  <span className="decision-date">{d.date}</span>
                  <span className="decision-action">{d.action}</span>
                </div>
                <div className="decision-rationale">{d.rationale}</div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

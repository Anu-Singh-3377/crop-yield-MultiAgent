import { Link } from 'react-router-dom';

const iconPaths = {
  leaf: <><path d="M20 4c-9 0-15 4-15 11a5 5 0 0 0 5 5c7 0 11-6 10-16Z" /><path d="M4 20c3-5 7-8 12-11" /></>,
  grid: <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /></>,
  field: <><path d="M3 20c4-5 7-5 11 0" /><path d="M10 20c4-7 7-7 11 0" /><path d="M3 16c3-3 5-4 8-4" /><path d="M5 10c3-3 6-4 10-4" /><path d="M15 5V3m-7 5V6m12 6v-2" /></>,
  health: <><path d="M20 7c-8 0-13 3-13 9a4 4 0 0 0 4 4c6 0 9-5 9-13Z" /><path d="M4 20c3-5 6-8 11-10" /><path d="M8 15h.01M13 12h.01M17 9h.01" /></>,
  weather: <><path d="M7 18h10a4 4 0 0 0 .4-8A6 6 0 0 0 6 11a3.5 3.5 0 0 0 1 7Z" /><path d="m9 21-1 2m6-2-1 2m6-2-1 2" /></>,
  yield: <><path d="M4 19V5m0 14h17" /><path d="m7 15 4-4 3 2 6-7" /><path d="M17 6h3v3" /></>,
  market: <><path d="M3 3v18h18" /><path d="m7 14 4-4 4 3 6-7" /><path d="M17 6h4v4" /></>,
  advisory: <><path d="M12 3 4 6v5c0 5 3.5 8.5 8 10 4.5-1.5 8-5 8-10V6l-8-3Z" /><path d="m9 12 2 2 4-4" /></>,
  history: <><path d="M3 12a9 9 0 1 0 2.6-6.4L3 8" /><path d="M3 3v5h5m4-1v5l3 2" /></>,
  menu: <><path d="M4 6h16M4 12h16M4 18h16" /></>,
  close: <><path d="m6 6 12 12M18 6 6 18" /></>,
  arrow: <><path d="M5 12h14" /><path d="m13 6 6 6-6 6" /></>,
  chevron: <><path d="m7 10 5 5 5-5" /></>,
  pin: <><path d="M20 10c0 5-8 11-8 11S4 15 4 10a8 8 0 1 1 16 0Z" /><circle cx="12" cy="10" r="2" /></>,
  cloud: <><path d="M7 18h10a4 4 0 0 0 .4-8A6 6 0 0 0 6 11a3.5 3.5 0 0 0 1 7Z" /><path d="M8 21h.01m4 0h.01m4 0h.01" /></>,
  upload: <><path d="M12 16V4m-4 4 4-4 4 4" /><path d="M4 16v4h16v-4" /></>,
  check: <><path d="m5 12 4 4L19 6" /></>,
  alert: <><path d="M12 3 2.8 19h18.4L12 3Z" /><path d="M12 9v4m0 3h.01" /></>,
};

export function Icon({ name, size = 20, className = '' }) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {iconPaths[name] || iconPaths.leaf}
    </svg>
  );
}

export function Button({ children, variant = 'primary', className = '', ...props }) {
  return <button className={`button button-${variant} ${className}`} {...props}>{children}</button>;
}

export function StatusBadge({ status = 'not-started' }) {
  const labels = {
    'not-started': 'Not started',
    submitted: 'Submitted',
    completed: 'Completed',
    failed: 'Needs attention',
    skipped: 'Not requested',
    available: 'Available',
    unavailable: 'Not available',
  };
  return <span className={`status-badge badge-${status}`}>{labels[status] || status}</span>;
}

export function AgentStep({
  step,
  icon,
  title,
  description,
  status,
  children,
  defaultOpen = false,
}) {
  return (
    <details className="agent-step" open={defaultOpen}>
      <summary className="agent-step-summary">
        <span className="step-number">{String(step).padStart(2, '0')}</span>
        <span className="agent-step-icon"><Icon name={icon} /></span>
        <span className="agent-step-copy">
          <strong>{title}</strong>
          <span>{description}</span>
        </span>
        {status && <StatusBadge status={status} />}
        <Icon name="chevron" className="step-chevron" />
      </summary>
      <div className="agent-step-content">{children}</div>
    </details>
  );
}

export function EmptyState({ title, description, action }) {
  return (
    <div className="empty-state">
      <span className="empty-state-icon"><Icon name="leaf" size={24} /></span>
      <h3>{title}</h3>
      <p>{description}</p>
      {action}
    </div>
  );
}

export function PageHeading({ eyebrow, title, description, action }) {
  return (
    <div className="page-heading">
      <div>
        {eyebrow && <span className="eyebrow">{eyebrow}</span>}
        <h1>{title}</h1>
        {description && <p>{description}</p>}
      </div>
      {action}
    </div>
  );
}

export function Field({ label, hint, children, className = '' }) {
  return (
    <label className={`field ${className}`}>
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

export function AgentResultStatus({ data }) {
  const names = [
    ['farmer_advisor', 'Farmer Advisor'],
    ['weather', 'Weather Agent'],
    ['disease', 'Crop Health Agent'],
    ['yield_prediction', 'Yield Prediction Agent'],
    ['market_researcher', 'Market Research Agent'],
    ['local_advisory', 'Sustainability & Advisory'],
  ];
  const executed = new Set(data?.executed_agents || []);
  const skipped = data?.skipped_agents || {};
  const errors = new Map((data?.agent_errors || []).map((error) => [error.agent, error]));
  return (
    <div className="agent-status-list">
      {names.map(([key, label]) => {
        const error = errors.get(key);
        const status = error ? 'failed' : executed.has(key) ? 'completed' : key in skipped ? 'skipped' : 'not-started';
        const detail = error?.message || skipped[key] || (status === 'completed' ? 'The backend reports this agent completed.' : 'No execution status was returned.');
        return (
          <div className="agent-status-row" key={key}>
            <span className={`agent-dot dot-${status}`} />
            <strong>{label}</strong>
            <StatusBadge status={status} />
            <p>{detail}</p>
          </div>
        );
      })}
    </div>
  );
}

export function LinkButton({ to, children, variant = 'primary', ...props }) {
  return <Link className={`button button-${variant}`} to={to} {...props}>{children}</Link>;
}

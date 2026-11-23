import { useState } from 'react';
import { NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import {
  AdvisoryPage,
  CropHealthPage,
  Dashboard,
  FarmAnalysis,
  FarmerRecommendationsPage,
  HistoryPage,
  MarketInsightsPage,
  WeatherPage,
  YieldPredictionPage,
} from './pages.jsx';
import { Icon } from './components/common.jsx';

const navigation = [
  { to: '/', label: 'Dashboard', icon: 'grid', end: true },
  { to: '/farm-analysis', label: 'Farm Analysis', icon: 'field' },
  { to: '/crop-health', label: 'Crop Health', icon: 'health' },
  { to: '/weather', label: 'Weather', icon: 'weather' },
  { to: '/yield-prediction', label: 'Yield Prediction', icon: 'yield' },
  { to: '/market-insights', label: 'Market Insights', icon: 'market' },
  { to: '/advisory', label: 'Advisory', icon: 'advisory' },
  { to: '/history', label: 'History', icon: 'history' },
];

function NavigationLink({ item, onClick }) {
  return (
    <NavLink
      to={item.to}
      end={item.end}
      onClick={onClick}
      className={({ isActive }) => `nav-link${isActive ? ' nav-link-active' : ''}`}
    >
      <Icon name={item.icon} size={19} />
      <span>{item.label}</span>
    </NavLink>
  );
}

function Sidebar({ mobileOpen, closeMenu }) {
  return (
    <>
      {mobileOpen && <button className="sidebar-backdrop" aria-label="Close navigation" onClick={closeMenu} />}
      <aside className={`sidebar${mobileOpen ? ' sidebar-open' : ''}`}>
        <div className="brand-lockup">
          <span className="brand-mark"><Icon name="leaf" size={23} /></span>
          <span><b>AGROAI</b><small>Farm intelligence</small></span>
        </div>
        <div className="sidebar-caption">WORKSPACE</div>
        <nav className="main-navigation" aria-label="Main navigation">
          {navigation.map((item) => <NavigationLink key={item.to} item={item} onClick={closeMenu} />)}
        </nav>
        <div className="sidebar-bottom">
          <div className="support-mark"><Icon name="advisory" size={18} /></div>
          <p><b>Evidence-led insights</b><span>Agent results are shown as returned by the backend.</span></p>
        </div>
      </aside>
    </>
  );
}

function Header({ onMenu }) {
  const location = useLocation();
  const current = navigation.find((item) => item.end ? location.pathname === item.to : location.pathname.startsWith(item.to));
  return (
    <header className="topbar">
      <button className="mobile-menu-button" onClick={onMenu} aria-label="Open navigation"><Icon name="menu" /></button>
      <div className="topbar-title"><span>AGROAI /</span> {current?.label || 'Dashboard'}</div>
      <div className="topbar-meta">Agricultural decision support</div>
    </header>
  );
}

export default function App() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  return (
    <div className="app-shell">
      <Sidebar mobileOpen={mobileOpen} closeMenu={() => setMobileOpen(false)} />
      <div className="app-main">
        <Header onMenu={() => setMobileOpen(true)} />
        <main className="content-area">
          <Routes>
            <Route path="/" element={<Dashboard analysis={analysis} />} />
            <Route path="/dashboard" element={<Navigate to="/" replace />} />
            <Route path="/farm-analysis" element={<FarmAnalysis analysis={analysis} onResult={setAnalysis} onStart={() => setAnalysis(null)} />} />
            <Route path="/crop-health" element={<CropHealthPage />} />
            <Route path="/weather" element={<WeatherPage />} />
            <Route path="/yield-prediction" element={<YieldPredictionPage />} />
            <Route path="/market-insights" element={<MarketInsightsPage />} />
            <Route path="/advisory" element={<AdvisoryPage analysis={analysis} />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/recommendations/:farmerId" element={<FarmerRecommendationsPage />} />
            <Route path="*" element={<Dashboard analysis={analysis} />} />
          </Routes>
          <footer className="app-footer">
            <span>AGROAI · Agricultural Decision Support</span>
            <span>Model outputs can be uncertain; confirm critical decisions locally.</span>
          </footer>
        </main>
      </div>
    </div>
  );
}

import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  AgentResultStatus,
  AgentStep,
  Button,
  EmptyState,
  Field,
  Icon,
  PageHeading,
  StatusBadge,
} from './components/common.jsx';
import { useFarmAnalysis } from './hooks/useFarmAnalysis.js';
import {
  getMarketAnalysis,
  getRecommendations,
  saveFarmerProfile,
} from './services/analysisApi.js';
import { detectDisease, getWeather, predictYield } from './services/agentApi.js';

const soilOptions = ['Clay', 'Sandy', 'Loamy', 'Silt'];
const waterOptions = ['High', 'Medium', 'Low'];

function listCrops(value) {
  return value.split(',').map((item) => item.trim()).filter(Boolean);
}

function profileFromForm(form) {
  return {
    name: form.name.trim(),
    location: form.location.trim(),
    farm_size: Number(form.farmSize),
    soil_type: form.soilType,
    water_availability: form.waterAvailability,
    preferred_crops: listCrops(form.crops),
    budget: Number(form.budget),
  };
}

function asNumber(value) {
  return value === '' || value === undefined ? undefined : Number(value);
}

function fileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || '');
      resolve(result.slice(result.indexOf(',') + 1));
    };
    reader.onerror = () => reject(new Error('The selected crop image could not be read.'));
    reader.readAsDataURL(file);
  });
}

function Value({ value, unit }) {
  if (value === undefined || value === null || value === '') return <>Not available</>;
  return <>{value}{unit ? ` ${unit}` : ''}</>;
}

function scoreText(value) {
  return typeof value === 'number' && value >= 0 && value <= 1
    ? `${Math.round(value * 100)}%`
    : value ?? 'Not available';
}

function humanize(value) {
  return String(value || '').replaceAll('_', ' ');
}

function InfoCard({ title, children, className = '' }) {
  return <article className={`info-card ${className}`}><h3>{title}</h3>{children}</article>;
}

function EmptyResult({ children = 'No result was returned by the backend.' }) {
  return <p className="empty-result">{children}</p>;
}

function ResultList({ items, render }) {
  if (!items?.length) return <EmptyResult>No items were returned by the agent.</EmptyResult>;
  return <ul className="result-list">{items.map((item, index) => <li key={item.id || item.crop || item.date || index}>{render(item)}</li>)}</ul>;
}

export function Dashboard({ analysis }) {
  const weatherRisks = analysis?.weather?.agricultural_risks || [];
  const marketItems = analysis?.market_analysis?.recommendations || [];
  const score = analysis?.sustainability_assessment?.[0]?.sustainability_score;
  const summary = [
    {
      icon: 'health',
      title: 'Farm health',
      value: analysis?.disease?.disease || (analysis ? 'Awaiting image result' : 'Awaiting analysis'),
      detail: analysis?.disease ? `Confidence ${Math.round(analysis.disease.confidence * 100)}%` : 'Based on submitted crop health data',
    },
    {
      icon: 'yield',
      title: 'Expected yield',
      value: analysis?.yield_prediction ? `${analysis.yield_prediction.predicted_yield} ${analysis.yield_prediction.unit}` : 'Not available',
      detail: analysis?.yield_prediction ? `Model ${analysis.yield_prediction.model_version}` : 'Requires a trained yield model',
    },
    {
      icon: 'weather',
      title: 'Weather risk',
      value: analysis?.weather ? (weatherRisks[0]?.severity || 'No risks reported') : 'Awaiting analysis',
      detail: analysis?.weather ? `${weatherRisks.length} risk indicator(s) returned` : 'Weather has not been requested',
    },
    {
      icon: 'market',
      title: 'Market outlook',
      value: marketItems[0]?.market_trend?.price_trend || 'Not available',
      detail: marketItems[0] ? `${marketItems[0].crop} · ${marketItems[0].market_trend.demand_level} demand` : 'No market results yet',
    },
    {
      icon: 'leaf',
      title: 'Sustainability',
      value: score === undefined ? 'Not available' : `${Math.round(score * 100)}%`,
      detail: score === undefined ? 'No Farmer Advisor score returned' : 'Farmer Advisor assessment',
    },
  ];

  return (
    <div className="page dashboard-page">
      <section className="hero-panel">
        <img src="/images/farm-fields.jpg" alt="Green agricultural fields under a wide open sky" fetchPriority="high" />
        <div className="hero-shade" />
        <div className="hero-copy">
          <span className="hero-kicker">AGRICULTURAL DECISION SUPPORT</span>
          <h1>Intelligent decisions<br />for better farming.</h1>
          <p>Bring together your farm details, local weather, crop health and market information to support your next decision.</p>
          <div className="hero-actions">
            <Link className="button button-primary" to="/farm-analysis">Start farm analysis <Icon name="arrow" size={18} /></Link>
            <Link className="button button-outline-light" to="/history">View analysis history</Link>
          </div>
        </div>
        <div className="hero-caption">Photo: Unsplash</div>
      </section>

      <PageHeading eyebrow="YOUR FARM OVERVIEW" title="Farm intelligence" description="Only backend results from your current session appear here. Run an assessment to see your farm-specific insights." />
      <div className="summary-grid">
        {summary.map((card) => (
          <article className="summary-card" key={card.title}>
            <span className="summary-icon"><Icon name={card.icon} /></span>
            <span className="summary-label">{card.title}</span>
            <strong>{card.value}</strong>
            <small>{card.detail}</small>
          </article>
        ))}
      </div>

      <section className="dashboard-lower">
        <InfoCard title={analysis ? 'Latest assessment' : 'No assessment yet'}>
          {analysis ? (
            <>
              <p>Crop: <strong><Value value={analysis.crop} /></strong></p>
              <p>Location: <strong><Value value={analysis.location} /></strong></p>
              <p>Agents reported: <strong>{analysis.executed_agents?.length || 0}</strong></p>
              <Link className="text-link" to="/farm-analysis">Review this assessment <Icon name="arrow" size={16} /></Link>
            </>
          ) : (
            <>
              <p className="muted">Your dashboard will show results after an AgroAI request returns data from the backend.</p>
              <Link className="text-link" to="/farm-analysis">Start a farm assessment <Icon name="arrow" size={16} /></Link>
            </>
          )}
        </InfoCard>
        <InfoCard title="How AgroAI works">
          <p className="muted">Submit farm information once. The existing orchestration graph selects applicable agents, reports unavailable services, and combines available outputs into an advisory.</p>
          <div className="source-chips"><span>Farm data</span><span>Agent outputs</span><span>Local advisory</span></div>
        </InfoCard>
      </section>
    </div>
  );
}

function resultStatus(data, agent, loading) {
  if (loading) return null;
  if (!data) return 'not-started';
  if ((data.agent_errors || []).some((error) => error.agent === agent)) return 'failed';
  if ((data.executed_agents || []).includes(agent)) return 'completed';
  if (data.skipped_agents && agent in data.skipped_agents) return 'skipped';
  return 'not-started';
}

function AnalysisResults({ data }) {
  const advisory = data.final_advisory;
  const weather = data.weather;
  const cropRows = data.sustainability_assessment || [];

  return (
    <section className="analysis-results" id="assessment-results">
      <div className="result-title-row">
        <div>
          <span className="eyebrow">BACKEND RESPONSE</span>
          <h2>AgroAI farm assessment</h2>
          <p>Values below are returned by the selected backend agents; unavailable outputs are not estimated.</p>
        </div>
        <StatusBadge status={data.agent_errors?.length ? 'failed' : 'completed'} />
      </div>
      <AgentResultStatus data={data} />
      <div className="results-grid">
        <InfoCard title="Farm & crop assessment">
          <p><b>Farmer:</b> <Value value={data.farmer_profile?.name} /></p>
          <p><b>Crop:</b> <Value value={data.crop} /></p>
          <p><b>Location:</b> <Value value={data.location} /></p>
          <ResultList items={cropRows} render={(item) => (
          <><b>{item.crop}</b> · Sustainability {scoreText(item.sustainability_score)} · Soil match {scoreText(item.soil_compatibility)}</>
          )} />
        </InfoCard>
        <InfoCard title="Weather observations">
          {weather ? (
            <>
              <p><b>{weather.location}</b></p>
              <div className="metric-row">
                <span><b><Value value={weather.temperature} unit="°C" /></b><small>Temperature</small></span>
                <span><b><Value value={weather.humidity} unit="%" /></b><small>Humidity</small></span>
                <span><b><Value value={weather.rainfall} unit="mm" /></b><small>Rainfall</small></span>
              </div>
              <h4>Agricultural risks</h4>
              <ResultList items={weather.agricultural_risks} render={(risk) => <><b>{humanize(risk.type)}</b> ({risk.severity}): {risk.description}</>} />
              <h4>Forecast</h4>
              <ResultList items={weather.forecast} render={(day) => `${day.date}: ${day.temperature_min ?? '–'}–${day.temperature_max ?? '–'}°C, ${day.rainfall ?? '–'} mm rain`} />
            </>
          ) : <EmptyResult>No weather output was returned.</EmptyResult>}
        </InfoCard>
        <InfoCard title="Crop health classification">
          {data.disease ? (
            <>
              <p><b>Crop:</b> <Value value={data.disease.crop} /></p>
              <p><b>Classifier result:</b> <Value value={data.disease.disease} /></p>
              <p><b>Confidence:</b> {Math.round(data.disease.confidence * 100)}%</p>
              <p><b>Risk:</b> <span className={`priority-badge priority-${String(data.disease.risk_level).toLowerCase()}`}>{humanize(data.disease.risk_level)} risk</span></p>
              {data.disease.confidence < 0.6 && <p className="inline-alert">Low-confidence prediction. Consider consulting an agricultural expert.</p>}
              <p className="muted">Model classification only; this is not a confirmed diagnosis.</p>
            </>
          ) : <EmptyResult>No disease classification was returned. A trained disease model and image are required.</EmptyResult>}
        </InfoCard>
        <InfoCard title="Yield model prediction">
          {data.yield_prediction ? (
            <>
              <div className="yield-number">{data.yield_prediction.predicted_yield} <small>{data.yield_prediction.unit}</small></div>
              <p>Crop: <b><Value value={data.yield_prediction.crop} /></b></p>
              <p>Model version: <b><Value value={data.yield_prediction.model_version} /></b></p>
              <p>Prediction range: <code>{JSON.stringify(data.yield_prediction.confidence_or_range)}</code></p>
              <h4>Model factors</h4>
              <ResultList items={data.yield_prediction.important_factors} render={(factor) => `${factor.factor}: ${factor.impact}`} />
            </>
          ) : <EmptyResult>No yield prediction was returned. Check model availability and agent status.</EmptyResult>}
        </InfoCard>
        <InfoCard title="Market research">
          {data.market_analysis ? (
            <ResultList items={data.market_analysis.recommendations} render={(item) => (
              <><b>{item.crop}</b> · {item.market_trend.demand_level} demand · {item.market_trend.price_trend} price trend · Current {item.market_trend.current_price} · Forecast {item.market_trend.predicted_price}</>
            )} />
          ) : <EmptyResult>No market analysis was returned.</EmptyResult>}
        </InfoCard>
        <InfoCard title="Sustainability & advisory">
          {advisory ? (
            <>
              <h4>Sustainable actions</h4>
              <ResultList items={advisory.sustainability_actions} render={(item) => <><b>{item.priority}:</b> {item.action} — {item.rationale}</>} />
              <h4>Warnings</h4>
              <ResultList items={advisory.warnings} render={(item) => item} />
            </>
          ) : <EmptyResult>No local advisory was returned.</EmptyResult>}
        </InfoCard>
        <InfoCard title="Priority actions" className="span-two">
          {advisory ? (
            <ResultList items={advisory.recommended_actions} render={(item) => (
              <div className="priority-action">
                <span className={`priority-badge priority-${item.priority}`}>{item.priority} priority</span>
                <span><b>{item.action}</b><br />{item.rationale}<small>{item.evidence?.join(' · ')}</small></span>
              </div>
            )} />
          ) : <EmptyResult>No recommendations were returned by Local Advisory.</EmptyResult>}
        </InfoCard>
      </div>
    </section>
  );
}

export function FarmAnalysis({ analysis, onResult, onStart }) {
  const { loading, error, runAnalysis } = useFarmAnalysis();
  const result = analysis;
  const [form, setForm] = useState({
    name: '', location: '', farmSize: '', soilType: '', waterAvailability: '',
    crops: '', crop: '', budget: '', soilPh: '', soilMoisture: '',
    latitude: '', longitude: '', temperature: '', rainfall: '',
    fertilizer: '', pesticide: '', historicalYield: '', sustainability: '',
    marketCrops: '',
  });
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [formError, setFormError] = useState('');
  const [saveState, setSaveState] = useState({ loading: false, message: '', farmerId: null });
  const completionFields = [form.name, form.location, form.farmSize, form.soilType, form.waterAvailability, form.budget, form.crop || form.crops].filter(Boolean).length;
  const completion = Math.round(completionFields / 7 * 100);

  useEffect(() => {
    if (!image) {
      setPreviewUrl('');
      return undefined;
    }
    const url = URL.createObjectURL(image);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [image]);

  const update = (key) => (event) => {
    setForm((current) => ({ ...current, [key]: event.target.value }));
  };
  const profile = useMemo(() => profileFromForm(form), [form]);
  const selectedCrop = form.crop.trim() || profile.preferred_crops[0] || '';

  async function handleSaveProfile() {
    setSaveState({ loading: true, message: '', farmerId: null });
    try {
      const saved = await saveFarmerProfile(profile);
      setSaveState({ loading: false, message: 'Farmer profile saved.', farmerId: saved.farmer_id });
    } catch (saveError) {
      setSaveState({ loading: false, message: saveError.message, farmerId: null });
    }
  }

  async function handleAnalyze(event) {
    event.preventDefault();
    setFormError('');
    if (!profile.name || !profile.location || !form.farmSize || !form.soilType || !form.waterAvailability || !form.budget || !selectedCrop) {
      setFormError('Complete the farmer name, location, farm area, soil, water availability, budget, and crop before analysis.');
      return;
    }
    if (!Number.isFinite(profile.farm_size) || profile.farm_size <= 0 || !Number.isFinite(profile.budget) || profile.budget < 0) {
      setFormError('Farm area must be greater than zero and the budget cannot be negative.');
      return;
    }
    if (form.soilPh !== '' && (Number(form.soilPh) < 0 || Number(form.soilPh) > 14)) {
      setFormError('Soil pH must be between 0 and 14.');
      return;
    }
    if (form.sustainability !== '' && (Number(form.sustainability) < 0 || Number(form.sustainability) > 100)) {
      setFormError('Sustainability score must be between 0 and 100.');
      return;
    }
    if ((form.latitude && !form.longitude) || (!form.latitude && form.longitude)) {
      setFormError('Enter both coordinates, or leave both blank.');
      return;
    }
    if (image && !image.type.startsWith('image/')) {
      setFormError('Choose a supported image file.');
      return;
    }
    if (image && image.size > 8 * 1024 * 1024) {
      setFormError('The crop image must be smaller than 8 MB.');
      return;
    }

    try {
      onStart?.();
      const payload = {
        intent: 'farm_assessment',
        farmer_profile: { ...profile, preferred_crops: profile.preferred_crops.length ? profile.preferred_crops : [selectedCrop] },
        crop: selectedCrop,
        market_crops: listCrops(form.marketCrops).length ? listCrops(form.marketCrops) : [selectedCrop],
        location: profile.location,
        farm_size: profile.farm_size,
        latitude: asNumber(form.latitude),
        longitude: asNumber(form.longitude),
        soil: {
          soil_type: profile.soil_type,
          soil_ph: asNumber(form.soilPh),
          soil_moisture: asNumber(form.soilMoisture),
          water_availability: profile.water_availability,
        },
        temperature_c: asNumber(form.temperature),
        rainfall_mm: asNumber(form.rainfall),
        fertilizer_usage_kg: asNumber(form.fertilizer),
        pesticide_usage_kg: asNumber(form.pesticide),
        historical_yield: asNumber(form.historicalYield),
        sustainability_score: asNumber(form.sustainability),
        image_crop: selectedCrop,
      };
      if (image) payload.image_base64 = await fileAsBase64(image);
      const response = await runAnalysis(payload);
      onResult(response);
      window.setTimeout(() => document.getElementById('assessment-results')?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50);
    } catch {
      // The hook keeps a user-safe error message for the page.
    }
  }

  const status = (agent) => resultStatus(result, agent, loading);

  return (
    <div className="page analysis-page">
      <PageHeading eyebrow="FARM-SPECIFIC WORKFLOW" title="Farm analysis" description="Work through each section, then submit one assessment to the existing AgroAI orchestration API. The vertical workflow shows returned agent status after the request completes." />
      <form onSubmit={handleAnalyze} noValidate>
        <div className="workflow-intro">
          <div><span className="workflow-line-marker"><Icon name="field" /></span><span><b>Your analysis workflow</b><small>Agents run only when required inputs are available.</small></span></div>
          <span className="completion-label">Profile completion <b>{completion}%</b></span>
          <div className="completion-track"><span style={{ width: `${completion}%` }} /></div>
        </div>

        <div className="workflow">
          <AgentStep step={1} icon="field" title="Farmer Profile" description="Tell AgroAI about your farm, resources, and primary crop." status={status('farmer_advisor')} defaultOpen>
            <div className="field-grid">
              <Field label="Farmer name" className="required"><input autoComplete="name" value={form.name} onChange={update('name')} required /></Field>
              <Field label="Farm location" className="required"><input autoComplete="address-level2" value={form.location} onChange={update('location')} required placeholder="Village, district, or town" /></Field>
              <Field label="Farm area (acres)" className="required"><input type="number" min="0.1" step="any" value={form.farmSize} onChange={update('farmSize')} required /></Field>
              <Field label="Soil type" className="required"><select value={form.soilType} onChange={update('soilType')} required><option value="">Select soil type</option>{soilOptions.map((soil) => <option key={soil}>{soil}</option>)}</select></Field>
              <Field label="Water availability" className="required"><select value={form.waterAvailability} onChange={update('waterAvailability')} required><option value="">Select availability</option>{waterOptions.map((water) => <option key={water}>{water}</option>)}</select></Field>
              <Field label="Farm budget (₹)" className="required"><input type="number" min="0" step="any" value={form.budget} onChange={update('budget')} required /></Field>
              <Field label="Preferred crops" hint="Separate crop names with commas."><input value={form.crops} onChange={update('crops')} placeholder="e.g. Wheat, rice" /></Field>
            </div>
            <div className="form-actions">
              <Button type="button" variant="secondary" onClick={handleSaveProfile} disabled={saveState.loading || !profile.name || !profile.location || !form.farmSize || !form.soilType || !form.waterAvailability || !form.budget}>
                {saveState.loading ? 'Saving profile…' : 'Save farmer profile'}
              </Button>
              {saveState.message && <span className={saveState.farmerId ? 'save-message' : 'form-error'} role="status">{saveState.message}</span>}
              {saveState.farmerId && <Link className="text-link" to={`/recommendations/${saveState.farmerId}?crops=${encodeURIComponent(profile.preferred_crops.join(','))}`}>View saved-profile recommendations <Icon name="arrow" size={15} /></Link>}
            </div>
          </AgentStep>

          <AgentStep step={2} icon="weather" title="Weather & Location" description="Use your farm location or provide coordinates for the weather service." status={status('weather')}>
            <p className="muted">Farm location: <b>{form.location || 'Enter this in Farm Profile'}</b>. Weather is retrieved by the backend when the farm assessment runs.</p>
            <div className="field-grid field-grid-compact">
              <Field label="Latitude (optional)"><input type="number" min="-90" max="90" step="any" value={form.latitude} onChange={update('latitude')} placeholder="e.g. 12.97" /></Field>
              <Field label="Longitude (optional)"><input type="number" min="-180" max="180" step="any" value={form.longitude} onChange={update('longitude')} placeholder="e.g. 77.59" /></Field>
            </div>
            {result?.weather && <p className="inline-success">Weather result received for {result.weather.location}.</p>}
          </AgentStep>

          <AgentStep step={3} icon="health" title="Crop Health" description="Optionally upload a crop image for the separate disease classifier." status={status('disease')}>
            <div className="upload-zone">
              <Icon name="upload" size={26} />
              <label htmlFor="farm-crop-image"><b>Upload or choose a crop image</b><span>Image files up to 8 MB · JPG, PNG, or WebP</span></label>
              <input id="farm-crop-image" type="file" accept="image/*" onChange={(event) => setImage(event.target.files?.[0] || null)} />
              {image && <div className="image-preview-card"><img src={previewUrl} alt="Preview of the selected crop" loading="lazy" /><div><b>{image.name}</b><small>{(image.size / 1024).toFixed(0)} KB</small><Button type="button" variant="text" onClick={() => { setImage(null); document.getElementById('farm-crop-image').value = ''; }}>Remove image</Button></div></div>}
            </div>
            <p className="muted">The classifier needs a trained artifact. If none is installed, its failure will be reported instead of showing an invented result.</p>
            {result?.disease && <p className="inline-success">Backend classification: {result.disease.disease}</p>}
          </AgentStep>

          <AgentStep step={4} icon="yield" title="Yield Prediction" description="Add available measurements; unsupported crop nutrition fields are not requested." status={status('yield_prediction')}>
            <div className="field-grid">
              <Field label="Soil pH" hint="Optional · 0 to 14"><input type="number" min="0" max="14" step="0.1" value={form.soilPh} onChange={update('soilPh')} /></Field>
              <Field label="Soil moisture" hint="Optional measured value"><input type="number" min="0" step="any" value={form.soilMoisture} onChange={update('soilMoisture')} /></Field>
              <Field label="Temperature (°C)"><input type="number" step="any" value={form.temperature} onChange={update('temperature')} /></Field>
              <Field label="Rainfall (mm)"><input type="number" min="0" step="any" value={form.rainfall} onChange={update('rainfall')} /></Field>
              <Field label="Fertilizer used (kg)"><input type="number" min="0" step="any" value={form.fertilizer} onChange={update('fertilizer')} /></Field>
              <Field label="Pesticide used (kg)"><input type="number" min="0" step="any" value={form.pesticide} onChange={update('pesticide')} /></Field>
              <Field label="Historical yield (tons)"><input type="number" min="0" step="any" value={form.historicalYield} onChange={update('historicalYield')} /></Field>
              <Field label="Sustainability score (0-100)"><input type="number" min="0" max="100" step="any" value={form.sustainability} onChange={update('sustainability')} /></Field>
            </div>
            <p className="muted">Crop: {selectedCrop || 'Select in Farmer Profile'} · Yield output is provided by the trained backend model.</p>
          </AgentStep>

          <AgentStep step={5} icon="market" title="Market Research" description="Request market analysis for your chosen crop or crop list." status={status('market_researcher')}>
            <Field label="Crops for market analysis" hint="Comma-separated; blank uses the selected crop."><input value={form.marketCrops} onChange={update('marketCrops')} placeholder={selectedCrop || 'e.g. Wheat, corn'} /></Field>
            <p className="muted">Region and farm size are taken from the Farmer Profile. Market prices and outlook are shown only as returned by the existing market service.</p>
          </AgentStep>

          <AgentStep step={6} icon="advisory" title="Sustainability & Local Advisory" description="The advisory combines available agent outputs and flags uncertainty." status={status('local_advisory')}>
            <p className="muted">No sustainability score or recommendation is generated in the browser. The backend Farmer Advisor and Local Advisory determine which assessments can be returned.</p>
            {result?.final_advisory && <p className="inline-success">{result.final_advisory.recommended_actions?.length || 0} advisory action(s) returned by Local Advisory.</p>}
          </AgentStep>
        </div>

        <div className="analysis-submit-wrap">
          {formError && <p className="form-error" role="alert">{formError}</p>}
          {error && <p className="form-error" role="alert">{error}</p>}
          {loading && (
            <div className="processing-note" role="status" aria-live="polite">
              <span className="spinner" />
              <span><b>Request sent to AgroAI</b><small>Waiting for the backend response. This API does not stream per-agent progress, so agent completion is shown only when the response arrives.</small></span>
            </div>
          )}
          <Button type="submit" className="analyse-cta" disabled={loading}>
            {loading ? 'Analysing your farm…' : 'Analyse my farm'} {!loading && <Icon name="arrow" size={19} />}
          </Button>
          <p className="cta-note">One request to the existing AgroAI orchestration endpoint · No predictions are generated in the browser</p>
        </div>
      </form>

      {result && <AnalysisResults data={result} />}
    </div>
  );
}

export function WeatherPage() {
  const [form, setForm] = useState({ location: '', latitude: '', longitude: '' });
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  async function submit(event) {
    event.preventDefault();
    setLoading(true); setError(''); setData(null);
    try {
      setData(await getWeather({ ...form, latitude: asNumber(form.latitude), longitude: asNumber(form.longitude) }));
    } catch (requestError) { setError(requestError.message); }
    finally { setLoading(false); }
  }
  return (
    <div className="page">
      <PageHeading eyebrow="LIVE PROVIDER DATA" title="Weather" description="Retrieve current conditions and agricultural risk indicators from the configured weather provider." />
      <InfoCard title="Find weather for your farm">
        <form className="inline-form" onSubmit={submit}>
          <Field label="Location"><input value={form.location} onChange={(event) => setForm({ ...form, location: event.target.value })} placeholder="Town or district" /></Field>
          <Field label="Latitude"><input type="number" min="-90" max="90" step="any" value={form.latitude} onChange={(event) => setForm({ ...form, latitude: event.target.value })} /></Field>
          <Field label="Longitude"><input type="number" min="-180" max="180" step="any" value={form.longitude} onChange={(event) => setForm({ ...form, longitude: event.target.value })} /></Field>
          <Button type="submit" disabled={loading}>{loading ? 'Loading…' : 'Get weather'}</Button>
        </form>
        {error && <p role="alert" className="form-error">{error}</p>}
        {loading && <p className="muted">Retrieving current conditions…</p>}
        {data && <WeatherResult data={data} />}
        {!data && !loading && !error && <EmptyState title="Weather not requested" description="Enter a location or both coordinates to request current conditions from the backend." />}
      </InfoCard>
    </div>
  );
}

function WeatherResult({ data }) {
  return (
    <div className="agent-result">
      <div className="metric-row">
        <span><b><Value value={data.temperature} unit="°C" /></b><small>Temperature</small></span>
        <span><b><Value value={data.humidity} unit="%" /></b><small>Humidity</small></span>
        <span><b><Value value={data.rainfall} unit="mm" /></b><small>Rainfall</small></span>
      </div>
      <p className="muted">Location: {data.location}</p>
      <h3>Agricultural risks</h3>
      <ResultList items={data.agricultural_risks} render={(risk) => <><b>{risk.type}</b> · {risk.severity}: {risk.description}</>} />
      <h3>Forecast</h3>
      <ResultList items={data.forecast} render={(day) => `${day.date} · ${day.temperature_min ?? '–'}–${day.temperature_max ?? '–'} °C · ${day.rainfall ?? '–'} mm rainfall`} />
    </div>
  );
}

export function CropHealthPage() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState('');
  const [crop, setCrop] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!image) { setPreview(''); return undefined; }
    const url = URL.createObjectURL(image); setPreview(url); return () => URL.revokeObjectURL(url);
  }, [image]);
  async function submit(event) {
    event.preventDefault();
    if (!image) { setError('Choose a crop image before analysis.'); return; }
    setLoading(true); setError(''); setResult(null);
    try { setResult(await detectDisease(image, crop)); }
    catch (requestError) { setError(requestError.message); }
    finally { setLoading(false); }
  }
  return (
    <div className="page">
      <PageHeading eyebrow="IMAGE CLASSIFICATION" title="Crop health" description="Upload a clear leaf or crop image. The backend must have a trained disease model artifact to return a classification." />
      <InfoCard title="Crop disease detection">
        <form onSubmit={submit} className="agent-form">
          <Field label="Crop (optional)"><input value={crop} onChange={(event) => setCrop(event.target.value)} placeholder="e.g. Tomato" /></Field>
          <Field label="Crop image"><input type="file" accept="image/*" onChange={(event) => { setImage(event.target.files?.[0] || null); setResult(null); }} required /></Field>
          {preview && <img className="agent-image-preview" src={preview} alt="Selected crop leaf preview" loading="lazy" />}
          <Button type="submit" disabled={loading || !image}>{loading ? 'Checking image…' : 'Analyze crop image'}</Button>
        </form>
        {error && <p role="alert" className="form-error">{error}</p>}
        {result && <div className="agent-result"><h3>{result.disease}</h3><p>Crop: <b>{result.crop}</b></p><p>Confidence: <b>{Math.round(result.confidence * 100)}%</b></p><p>Risk level: <span className={`priority-badge priority-${String(result.risk_level).toLowerCase()}`}>{humanize(result.risk_level)} risk</span></p>{result.confidence < 0.6 && <p className="inline-alert">Low-confidence prediction. Consider consulting an agricultural expert.</p>}<p className="muted">Classifier output is not a confirmed diagnosis.</p></div>}
        {!result && !loading && !error && <EmptyState title="No crop result yet" description="A prediction will appear here only after the backend returns one." />}
      </InfoCard>
    </div>
  );
}

export function YieldPredictionPage() {
  const [form, setForm] = useState({ crop: '', location: '', soil_ph: '', soil_moisture: '', temperature_c: '', rainfall_mm: '', fertilizer_usage_kg: '', pesticide_usage_kg: '', sustainability_score: '', historical_yield: '' });
  const [result, setResult] = useState(null); const [loading, setLoading] = useState(false); const [error, setError] = useState('');
  async function submit(event) {
    event.preventDefault(); setLoading(true); setError(''); setResult(null);
    const request = Object.fromEntries(Object.entries(form).map(([key, value]) => [key, key === 'crop' || key === 'location' ? value || undefined : asNumber(value)]));
    try { setResult(await predictYield(request)); }
    catch (requestError) { setError(requestError.message); }
    finally { setLoading(false); }
  }
  const change = (key) => (event) => setForm((current) => ({ ...current, [key]: event.target.value }));
  return (
    <div className="page">
      <PageHeading eyebrow="TRAINED MODEL OUTPUT" title="Yield prediction" description="Submit only the features supported by the existing yield model. Empty optional values are left for backend preprocessing." />
      <InfoCard title="Yield model inputs">
        <form className="field-grid" onSubmit={submit}>
          <Field label="Crop" className="required"><input value={form.crop} onChange={change('crop')} required /></Field>
          <Field label="Location"><input value={form.location} onChange={change('location')} /></Field>
          <Field label="Soil pH"><input type="number" min="0" max="14" step="any" value={form.soil_ph} onChange={change('soil_ph')} /></Field>
          <Field label="Soil moisture"><input type="number" min="0" step="any" value={form.soil_moisture} onChange={change('soil_moisture')} /></Field>
          <Field label="Temperature (°C)"><input type="number" step="any" value={form.temperature_c} onChange={change('temperature_c')} /></Field>
          <Field label="Rainfall (mm)"><input type="number" min="0" step="any" value={form.rainfall_mm} onChange={change('rainfall_mm')} /></Field>
          <Field label="Fertilizer usage (kg)"><input type="number" min="0" step="any" value={form.fertilizer_usage_kg} onChange={change('fertilizer_usage_kg')} /></Field>
          <Field label="Pesticide usage (kg)"><input type="number" min="0" step="any" value={form.pesticide_usage_kg} onChange={change('pesticide_usage_kg')} /></Field>
          <Field label="Sustainability score (0-100)"><input type="number" min="0" max="100" step="any" value={form.sustainability_score} onChange={change('sustainability_score')} /></Field>
          <Field label="Historical yield (tons)"><input type="number" min="0" step="any" value={form.historical_yield} onChange={change('historical_yield')} /></Field>
          <div className="field-span"><Button type="submit" disabled={loading}>{loading ? 'Predicting…' : 'Request yield prediction'}</Button></div>
        </form>
        {error && <p role="alert" className="form-error">{error}</p>}
        {result && <div className="agent-result"><div className="yield-number">{result.predicted_yield} <small>{result.unit}</small></div><p>Model version: {result.model_version}</p><p>Range: <code>{JSON.stringify(result.confidence_or_range)}</code></p><h3>Important factors</h3><ResultList items={result.important_factors} render={(factor) => `${factor.factor} · impact ${factor.impact}`} /></div>}
      </InfoCard>
    </div>
  );
}

export function MarketInsightsPage() {
  const [region, setRegion] = useState(''); const [crops, setCrops] = useState(''); const [result, setResult] = useState(null); const [loading, setLoading] = useState(false); const [error, setError] = useState('');
  async function submit(event) {
    event.preventDefault(); const cropList = listCrops(crops);
    if (!region.trim()) { setError('Enter a region to request market research.'); return; }
    setLoading(true); setError(''); setResult(null);
    try { setResult(await getMarketAnalysis(region.trim(), cropList)); }
    catch (requestError) { setError(requestError.message); }
    finally { setLoading(false); }
  }
  return (
    <div className="page">
      <PageHeading eyebrow="REGIONAL MARKET DATA" title="Market insights" description="Request the existing market research endpoint. Results below are returned by the backend." />
      <InfoCard title="Market research request">
        <form className="inline-form" onSubmit={submit}>
          <Field label="Region"><input value={region} onChange={(event) => setRegion(event.target.value)} required placeholder="District or market region" /></Field>
          <Field label="Crop list" hint="Comma-separated; blank uses backend defaults."><input value={crops} onChange={(event) => setCrops(event.target.value)} /></Field>
          <Button type="submit" disabled={loading}>{loading ? 'Loading…' : 'Request market insights'}</Button>
        </form>
        {error && <p role="alert" className="form-error">{error}</p>}
        {result && <div className="agent-result"><p>Region: <b>{result.region}</b></p><ResultList items={result.market_analysis} render={(item) => <><b>{item.crop}</b> · {item.market_trend.demand_level} demand · {item.market_trend.price_trend} price trend · Current price {item.market_trend.current_price} · Predicted price {item.market_trend.predicted_price} · Score {item.recommendation_score}</>} /></div>}
      </InfoCard>
    </div>
  );
}

export function AdvisoryPage({ analysis }) {
  const advisory = analysis?.final_advisory;
  return (
    <div className="page">
      <PageHeading eyebrow="EVIDENCE-BASED GUIDANCE" title="Local advisory" description="Recommendations are assembled from actual upstream agent outputs and include uncertainty warnings." />
      {advisory ? (
        <div className="results-grid">
          <InfoCard title="Recommended actions"><ResultList items={advisory.recommended_actions} render={(item) => <><b>{item.priority} priority:</b> {item.action}<p>{item.rationale}</p>{item.evidence?.length > 0 && <small>Evidence: {item.evidence.join(', ')}</small>}</>} /></InfoCard>
          <InfoCard title="Sustainability actions"><ResultList items={advisory.sustainability_actions} render={(item) => <><b>{item.action}</b><p>{item.rationale}</p></>} /></InfoCard>
          <InfoCard title="Warnings"><ResultList items={advisory.warnings} render={(item) => item} /></InfoCard>
        </div>
      ) : <EmptyState title="No advisory in this session" description="Run a farm analysis first. The advisory page uses only the response returned by the backend." action={<Link className="button button-primary" to="/farm-analysis">Start farm analysis</Link>} />}
    </div>
  );
}

export function HistoryPage() {
  return (
    <div className="page">
      <PageHeading eyebrow="SAVED ASSESSMENTS" title="Analysis history" description="Past assessments can be listed here once the backend provides a history endpoint." />
      <EmptyState title="Analysis history is not available yet" description="The current backend does not expose a route that saves and lists AgroAI runs. This page does not create local or sample history." action={<Link className="button button-secondary" to="/farm-analysis">Start a new assessment</Link>} />
    </div>
  );
}

export function FarmerRecommendationsPage() {
  const match = window.location.pathname.match(/\/recommendations\/(\d+)/);
  const farmerId = match?.[1];
  const crops = new URLSearchParams(window.location.search).get('crops')?.split(',').filter(Boolean) || [];
  const [data, setData] = useState(null); const [error, setError] = useState(''); const [loading, setLoading] = useState(true);
  useEffect(() => {
    if (!farmerId) { setError('Farmer profile identifier is missing.'); setLoading(false); return; }
    getRecommendations(farmerId, crops).then(setData).catch((requestError) => setError(requestError.message)).finally(() => setLoading(false));
  }, [farmerId]);
  return (
    <div className="page">
      <PageHeading eyebrow="SAVED FARM PROFILE" title="Farmer recommendations" description="Recommendations returned by the existing backend route." />
      {loading && <p className="muted">Loading saved-profile recommendations…</p>}
      {error && <p role="alert" className="form-error">{error}</p>}
      {data && <div className="results-grid"><InfoCard title={`Farmer #${data.farmer_id}`}><p>Generated: {data.timestamp}</p></InfoCard><InfoCard title="Recommendations"><ResultList items={data.recommendations} render={(item) => <><b>{item.crop}</b> · Sustainability {item.sustainability_score} · Market score {item.market_analysis?.recommendation_score}</>} /></InfoCard></div>}
      <Link className="text-link" to="/farm-analysis">Return to farm analysis <Icon name="arrow" size={15} /></Link>
    </div>
  );
}

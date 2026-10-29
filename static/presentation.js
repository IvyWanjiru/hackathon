/* Panel controls use the existing APIs, always scoped to the demo workspace. */
(() => {
  'use strict';
  const config = JSON.parse(document.getElementById('presentation-config').textContent);
  const $ = id => document.getElementById(id);
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const steps = ['story', 'capture', 'evidence', 'action', 'model', 'pilot'];
  const labels = ['The problem', 'Live capture', 'Customer evidence', 'Take action', 'The ML model', 'Pilot plan'];
  const shortLabels = ['Opening', 'Capture', 'Evidence', 'Action', 'ML', 'Pilot'];
  const notes = [
    ['0:00–0:25', 'The missing side of retail demand', 'A sale tells a retailer what sold. It does not explain the customer who asked for a product that was unavailable, too expensive or too late. GhostDemand keeps that missed request and turns it into evidence the team can act on.', 'Start the demo. All examples are synthetic; channel names are source labels.'],
    ['0:25–1:00', 'Capture, then review', 'Here is a request for two earbuds that are sold out. The backend extracts a draft: product, quantity, budget and blocker. I can correct it before saving. The budget is an expression of interest; it is not revenue.', 'Extract the signal → check the fields → tick the review box → save. Repeating the same sample does not create another request.'],
    ['1:00–1:25', 'Evidence supports the decision', 'The workspace groups related requests and preserves the original wording. We can see how many requests, known references and channel labels support this opportunity. Transparent business rules suggest a next step based on the recorded blockers.', 'Point to the counts, source messages and explanation. Counts reflect the current demo database.'],
    ['1:25–1:50', 'Assign the next step', 'An insight is useful when someone owns the response. I assign this sample request to a team member with a date and an action. The workspace also records outcomes. Real recovered revenue needs an order link; this demonstration shows synthetic, team-entered amounts.', 'Save the sample follow-up, then point to the confirmation. This does not send a customer message.'],
    ['1:50–2:35', 'Explain the model honestly', 'We convert the request text into TF-IDF features: weighted character fragments. A logistic regression classifier learns weights for six reasons a purchase failed. Its scores guide the blocker suggestion; a person reviews it. Product parsing and prioritisation use separate rules. The accompanying notebook achieved macro F1 of 0.860 on 21 synthetic test requests, but accepted none automatically at its selected threshold. This is a workflow experiment, not evidence of real-customer accuracy.', 'Separate the running baseline from the notebook candidate. SetFit with MiniLM is an optional next experiment, not the model serving this demo.'],
    ['2:35–3:00', 'A measured retail pilot', 'Our next step is one retailer: connect the actual channels, label real unmet requests, and evaluate by channel, language and time. Then measure useful actions and order-verified recoveries. The contribution is the complete loop from missed request to evidence, ownership and learning.', 'Finish here or open the workspace for questions. Download the notes and notebook for technical discussion.']
  ];
  const scenarios = {
    stock: 'The requested 2 wireless earbuds are sold out. Total budget KES 6,000. The customer needs them today.',
    price: 'The customer declined the quote for 2 wireless earbuds because the price is too high. Total budget KES 4,000.',
    delivery: 'The customer needs 2 wireless earbuds today, but delivery takes a week. Total budget KES 6,000.'
  };
  const state = {ready:false, setup:null, step:0, draft:null, saved:null, evidence:null, target:null, busy:false};
  let toastTimeout;
  async function api(path, options = {}) {
    const response = await fetch(path, {...options, headers:{'Content-Type':'application/json', ...options.headers, 'X-Ghost-Demo':'1'}});
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `The demo request failed (${response.status}).`);
    return data;
  }
  function error(err) {
    $('presentation-error').querySelector('span').textContent = err.message || 'The demo could not load. Try again.';
    $('presentation-error').hidden = false;
  }
  function toast(message) {
    clearTimeout(toastTimeout);
    $('presentation-toast').textContent = message;
    $('presentation-toast').hidden = false;
    toastTimeout = setTimeout(() => { $('presentation-toast').hidden = true; }, 5000);
  }
  async function busy(button, task) {
    if (button.disabled) return;
    button.disabled = true;
    button.setAttribute('aria-busy', 'true');
    $('presentation-error').hidden = true;
    try { await task(); } catch (err) { error(err); }
    finally { button.disabled = false; button.removeAttribute('aria-busy'); }
  }
  function money(values) {
    return Object.entries(values || {}).map(([currency, amount]) => `${currency} ${Number(amount).toLocaleString(undefined, {maximumFractionDigits:2})}`).join(' · ') || 'No amount recorded';
  }
  function renderModel(model) {
    $('panel-model-status').textContent = model.ready && model.enabled ? 'LOCAL CLASSIFIER READY' : 'RULES FALLBACK · MODEL UNAVAILABLE';
    const meta = model.metadata || {};
    $('panel-model-version').textContent = model.ready
      ? `Serving version ${meta.version || 'saved baseline'} · ${meta.examples ?? '—'} training examples · score threshold ${model.threshold}. Scores are not calibrated probabilities. Reviewed labels enter a future retraining run.`
      : 'The classifier is unavailable. Rules still produce an editable draft; no ML prediction is claimed.';
  }
  async function refreshCase() {
    const [insights, model] = await Promise.all([api('/api/insights?days=all'), api('/api/model')]);
    const k = insights.kpis;
    $('panel-total').textContent = k.signals;
    $('panel-opportunities').textContent = k.active_opportunities;
    $('panel-recoveries').textContent = k.recovered;
    $('panel-recovered-value').textContent = money(k.recovered_value);
    renderModel(model);
    const opportunity = (state.saved && insights.demand.find(row => row.signal_ids.includes(state.saved.id)))
      || insights.demand.find(row => /earbud/i.test(row.product)) || insights.demand[0];
    if (!opportunity) throw new Error('No sample opportunity is available. Retry the sample setup.');
    const evidence = await api(`/api/opportunities/evidence?days=all&key=${encodeURIComponent(opportunity.key)}`);
    state.evidence = evidence;
    $('case-product').textContent = opportunity.product;
    document.querySelector('.large-product-icon').textContent = opportunity.product.split(/\s+/).slice(0,2).map(word => word[0]).join('').toUpperCase();
    $('case-priority').textContent = `${opportunity.priority} priority`;
    $('case-requests').textContent = opportunity.count;
    $('case-customers').textContent = opportunity.customers;
    $('case-channels').textContent = Object.keys(opportunity.sources).length;
    $('case-action').textContent = opportunity.suggested_action;
    $('case-detail').textContent = opportunity.action_detail;
    $('case-basis').textContent = opportunity.action_basis;
    const sorted = [...evidence.items].sort((a,b) => Number(b.id === state.saved?.id) - Number(a.id === state.saved?.id));
    $('panel-evidence-list').innerHTML = sorted.slice(0, 3).map(row => `<article class="panel-evidence-item"><div><span class="present-tag">${escape(config.sources[row.source] || row.source)}</span><span>${escape(row.customer || 'Unidentified reference')}</span></div><blockquote>“${escape(row.raw)}”</blockquote><small>${escape(config.reasons[row.effective_reason || row.reviewed_reason || row.reason] || 'Review blocker')} · Request #${row.id}</small></article>`).join('');
    $('evidence-coverage').textContent = `Showing ${Math.min(3, evidence.items.length)} of ${evidence.items.length} synthetic requests for this product. Known references are not verified unique people.`;
    state.target = state.saved || evidence.items.find(row => row.external_id === 'demo-v1-0' && row.status !== 'recovered') || evidence.items.find(row => row.status !== 'recovered');
    $('panel-action-context').textContent = state.target ? `Request #${state.target.id} · ${state.target.product} · ${config.sources[state.target.source] || state.target.source}` : 'No open sample request is available.';
    $('panel-assign').disabled = !state.target;
  }
  async function ensureDemo() {
    if (state.ready) return;
    if (state.setup) return state.setup;
    $('demo-status').textContent = 'Preparing synthetic workspace…';
    state.setup = (async () => {
      try {
        await api('/api/demo/seed', {method:'POST', body:'{}'});
        await refreshCase();
        state.ready = true;
        $('demo-status').textContent = 'Demo workspace · synthetic data';
        $('presentation-error').hidden = true;
      } catch (err) {
        $('demo-status').textContent = 'Setup needs attention';
        throw err;
      } finally { state.setup = null; }
    })();
    return state.setup;
  }
  function setStep(key, focus = true) {
    clearTimeout(toastTimeout);
    $('presentation-toast').hidden = true;
    const index = steps.indexOf(key);
    state.step = index < 0 ? 0 : index;
    document.querySelectorAll('[data-scene]').forEach(scene => { scene.hidden = scene.dataset.scene !== steps[state.step]; });
    $('chapter-count').textContent = `${String(state.step + 1).padStart(2, '0')} / 06`;
    $('chapter-progress').style.width = `${(state.step + 1) / steps.length * 100}%`;
    $('chapter-back').disabled = state.step === 0;
    const nextLabel = state.step === 5 ? 'Back to opening ↺' : `${labels[state.step + 1]} →`;
    const nextShort = state.step === 5 ? 'Restart ↺' : `${shortLabels[state.step + 1]} →`;
    $('chapter-next').innerHTML = `<span class="present-next-full">${nextLabel}</span><span class="present-next-short">${nextShort}</span>`;
    $('chapter-next').setAttribute('aria-label', nextLabel);
    const note = notes[state.step];
    $('notes-timing').textContent = note[0];
    $('notes-heading').textContent = note[1];
    $('notes-script').textContent = note[2];
    $('notes-cue').textContent = note[3];
    if (focus) document.querySelector(`[data-scene="${steps[state.step]}"] h1`).focus({preventScroll:true});
    window.scrollTo({top:0, behavior:'instant'});
    if (state.step > 0) ensureDemo().catch(error);
  }
  function go(index) {
    const key = steps[Math.max(0, Math.min(steps.length - 1, index))];
    if (location.hash === `#${key}`) setStep(key); else location.hash = key;
  }
  window.addEventListener('hashchange', () => {
    const chapter = location.hash.slice(1);
    if (steps.includes(chapter)) setStep(chapter);
  });
  $('start-walkthrough').addEventListener('click', () => go(1));
  $('chapter-next').addEventListener('click', () => go((state.step + 1) % steps.length));
  $('chapter-back').addEventListener('click', () => go(state.step - 1));
  $('retry-demo').addEventListener('click', () => busy($('retry-demo'), async () => { state.ready = false; await ensureDemo(); }));
  function invalidateDraft() {
    state.draft = null;
    $('panel-draft-form').hidden = true;
    $('panel-draft-empty').hidden = false;
    $('extraction-status').textContent = 'Ready to preview';
    $('panel-capture-result').hidden = true;
    $('panel-verified').checked = false;
    $('panel-save').disabled = true;
  }
  $('panel-text').addEventListener('input', invalidateDraft);
  $('panel-source').addEventListener('change', invalidateDraft);
  document.querySelectorAll('[data-scenario]').forEach(button => button.addEventListener('click', () => {
    document.querySelectorAll('[data-scenario]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    $('panel-text').value = scenarios[button.dataset.scenario];
    invalidateDraft();
  }));
  $('panel-capture-form').addEventListener('submit', event => {
    event.preventDefault();
    busy($('panel-preview'), async () => {
      await ensureDemo();
      const text = $('panel-text').value.trim(), source = $('panel-source').value;
      const bytes = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(`${source}\n${text}`));
      const fingerprint = Array.from(new Uint8Array(bytes), b => b.toString(16).padStart(2,'0')).join('').slice(0,24);
      const draft = await api('/api/signals/preview', {method:'POST', body:JSON.stringify({text, source, customer:'Panel sample', currency:config.defaultCurrency, external_id:`panel-v1-${fingerprint}`})});
      // Ignore a response if the presenter changed the input while it was loading.
      if ($('panel-text').value.trim() !== text || $('panel-source').value !== source) return;
      state.draft = draft;
      const f = draft.fields;
      $('panel-product').value = f.product || '';
      $('panel-qty').value = f.qty ?? 1;
      $('panel-budget').value = f.budget ?? '';
      $('panel-currency').value = f.currency || config.defaultCurrency;
      $('panel-reason').value = Object.hasOwn(config.reasons, f.reason) ? f.reason : '';
      $('panel-reason').required = true;
      $('panel-verified').checked = false;
      $('panel-save').disabled = true;
      $('panel-draft-empty').hidden = true;
      $('panel-draft-form').hidden = false;
      $('panel-capture-result').hidden = true;
      $('extraction-status').textContent = f.reason_engine === 'local_ml' ? 'ML suggestion · review' : 'Rules draft · review';
      $('panel-warnings').innerHTML = draft.warnings.map(warning => `<li>${escape(warning)}</li>`).join('');
      $('panel-warnings').hidden = !draft.warnings.length;
    });
  });
  $('panel-verified').addEventListener('change', () => { $('panel-save').disabled = !$('panel-verified').checked; });
  $('panel-draft-form').addEventListener('input', event => {
    if (event.target.id !== 'panel-verified') {
      $('panel-verified').checked = false;
      $('panel-save').disabled = true;
    }
  });
  $('panel-draft-form').addEventListener('submit', event => {
    event.preventDefault();
    if (!state.draft || !$('panel-verified').checked) return;
    busy($('panel-save'), async () => {
      const {fields, warnings, ...identity} = state.draft;
      const row = await api('/api/signals', {method:'POST', body:JSON.stringify({...identity, verified:true, fields:{product:$('panel-product').value.trim(), qty:Number($('panel-qty').value), budget:$('panel-budget').value === '' ? null : Number($('panel-budget').value), currency:$('panel-currency').value, reason:$('panel-reason').value}})});
      state.saved = row;
      $('panel-action-result').hidden = true;
      $('panel-capture-result').textContent = row.deduplicated ? `Request #${row.id} already exists. No duplicate was added; open the workspace to edit the saved record.` : `Reviewed request #${row.id} saved to the demo workspace. Continue to Evidence →`;
      $('panel-capture-result').hidden = false;
      $('extraction-status').textContent = row.deduplicated ? 'Sample already saved' : 'Reviewed & saved';
      $('panel-verified').checked = false;
      await refreshCase();
    }).finally(() => { $('panel-save').disabled = !$('panel-verified').checked; });
  });
  $('panel-action-form').addEventListener('submit', event => {
    event.preventDefault();
    busy($('panel-assign'), async () => {
      await ensureDemo();
      if (!state.target) throw new Error('Capture a sample request before assigning a follow-up.');
      const row = await api(`/api/signals/${state.target.id}`, {method:'PATCH', body:JSON.stringify({owner:$('panel-owner').value.trim(), follow_up_on:$('panel-due').value, next_action:$('panel-next-action').value.trim(), status:'follow_up'})});
      if (state.saved?.id === row.id) state.saved = row;
      $('panel-action-result').textContent = `Request #${row.id} assigned to ${row.owner} · due ${row.follow_up_on}. ${row.next_action}`;
      $('panel-action-result').hidden = false;
      await refreshCase();
      toast('Sample follow-up saved. Open the queue to see the record.');
    });
  });
  function toggleNotes(force) {
    const open = force ?? $('presenter-notes').hidden;
    $('presenter-notes').hidden = !open;
    $('notes-toggle').setAttribute('aria-expanded', String(open));
  }
  $('notes-toggle').addEventListener('click', () => toggleNotes());
  $('notes-close').addEventListener('click', () => { toggleNotes(false); $('notes-toggle').focus(); });
  async function fullscreen() {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else if (document.documentElement.requestFullscreen) await document.documentElement.requestFullscreen();
      else toast('Use your browser’s fullscreen control for a larger presentation.');
    } catch (_) { toast('Fullscreen is unavailable here. Use the browser window controls.'); }
  }
  $('fullscreen').addEventListener('click', fullscreen);
  document.addEventListener('fullscreenchange', () => $('fullscreen').setAttribute('aria-label', document.fullscreenElement ? 'Exit fullscreen' : 'Enter fullscreen'));
  document.addEventListener('keydown', event => {
    if (event.altKey || event.ctrlKey || event.metaKey || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName) || event.target.isContentEditable) return;
    if (event.key === 'ArrowRight') { event.preventDefault(); go(Math.min(5,state.step + 1)); }
    if (event.key === 'ArrowLeft') { event.preventDefault(); go(state.step - 1); }
    if (event.key.toLowerCase() === 'n') toggleNotes();
    if (event.key.toLowerCase() === 'f') fullscreen();
    if (event.key === 'Escape') toggleNotes(false);
  });
  let remaining = 180000, started = null, interval = null;
  function drawTime() {
    const left = Math.max(0, remaining - (started === null ? 0 : Date.now() - started));
    const seconds = Math.ceil(left / 1000);
    $('rehearsal-timer').textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2,'0')}`;
    if (!left && started !== null) {
      remaining = 0; started = null; clearInterval(interval); updateTimerButton();
      toast('Three minutes reached. Time to wrap up or take questions.');
    }
  }
  function updateTimerButton() {
    $('timer-toggle').textContent = started === null ? '▷' : 'Ⅱ';
    $('timer-toggle').setAttribute('aria-pressed', String(started !== null));
    $('timer-toggle').setAttribute('aria-label', started === null ? 'Start three-minute rehearsal timer' : 'Pause rehearsal timer');
  }
  $('timer-toggle').addEventListener('click', () => {
    if (started === null) { if (!remaining) remaining = 180000; started = Date.now(); interval = setInterval(drawTime, 250); }
    else { remaining = Math.max(0, remaining - (Date.now() - started)); started = null; clearInterval(interval); }
    updateTimerButton(); drawTime();
  });
  $('timer-reset').addEventListener('click', () => { remaining = 180000; started = null; clearInterval(interval); updateTimerButton(); drawTime(); });
  setStep(location.hash.slice(1) || 'story', false);
})();

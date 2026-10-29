'use strict';
const cfg = JSON.parse(document.getElementById('app-config').textContent);
const $ = id => document.getElementById(id);
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = value => Number(value || 0).toLocaleString(undefined, {maximumFractionDigits:2});
const money = totals => Object.entries(totals || {}).map(([currency,value]) => `${currency === 'UNKNOWN' ? 'Unspecified currency' : currency} ${number(value)}`).join(' · ');
const date = value => new Date(value).toLocaleDateString(undefined, {month:'short',day:'numeric'});
const state = {page:'overview',filters:{q:'',days:'30',source:'',status:'',reason:'',review:''},signalPage:1,insights:null,signals:new Map(),draft:null,imports:[],requestVersion:0,signalRequestVersion:0,captureVersion:0,followupsVersion:0,followups:{status:'active',q:'',owner:'',urgency:'all',due:'all',page:1}};
let toastTimer, filterTimer, recognition;

function toast(message, error=false) {
  clearTimeout(toastTimer);
  $('toast').textContent = message;
  $('toast').classList.toggle('error',error);
  $('toast').hidden = false;
  toastTimer = setTimeout(() => {$('toast').hidden=true;}, error ? 12000 : 6500);
}
async function api(path, options={}) {
  const headers = {...(cfg.demo ? {'X-Ghost-Demo':'1'} : {}),...options.headers};
  if (!(options.body instanceof FormData)) headers['Content-Type'] = 'application/json';
  const response = await fetch(path, {...options, headers});
  let payload;
  try { payload = await response.json(); } catch { throw new Error('The server returned an unexpected response. Please try again.'); }
  if (!response.ok) throw new Error(payload.error || 'The request failed. Please try again.');
  return payload;
}
async function busy(button, label, action) {
  const original = button.textContent;
  button.disabled = true; button.textContent = label;
  try { return await action(); }
  finally { button.disabled = false; button.textContent = original; }
}
function query(extra={}) {
  const params = new URLSearchParams({...state.filters,...extra});
  for (const [key,value] of [...params]) if (!value) params.delete(key);
  return params.toString();
}
function filterMarkup(prefix) {
  const select = (key,label,options) => `<div><label for="${prefix}-${key}">${label}</label><select id="${prefix}-${key}" data-filter="${key}">${options.map(([value,text])=>`<option value="${esc(value)}">${esc(text)}</option>`).join('')}</select></div>`;
  return `<div><label for="${prefix}-q">Find a product, message or customer</label><input type="search" id="${prefix}-q" data-filter="q" placeholder="Search customer demand…" maxlength="150"></div>`
    + select('days','Time period',[['7','Last 7 days'],['30','Last 30 days'],['90','Last 90 days'],['all','All time']])
    + select('source','Channel',[['','All channels'],...Object.entries(cfg.sources)])
    + select('status','Follow-up',[['','All statuses'],...Object.entries(cfg.statuses)])
    + select('reason','Blocker',[['','All blockers'],...Object.entries(cfg.reasons)])
    + select('review','Review',[['','All requests'],['needed','Needs review']]);
}
function syncFilters() {
  document.querySelectorAll('[data-filter]').forEach(element => {if (element !== document.activeElement) element.value=state.filters[element.dataset.filter];});
}
for (const prefix of ['overview','inbox']) $(prefix+'-filters').innerHTML=filterMarkup(prefix);
syncFilters();
document.querySelectorAll('.filterbar').forEach(bar => {
  bar.addEventListener('input',event => {
    if (event.target.dataset.filter !== 'q') return;
    state.filters.q=event.target.value;state.signalPage=1;syncFilters();
    clearTimeout(filterTimer);filterTimer=setTimeout(refresh,250);
  });
  bar.addEventListener('change',event => {
    const key=event.target.dataset.filter;if(!key||key==='q')return;
    state.filters[key]=event.target.value;state.signalPage=1;syncFilters();refresh();
  });
});
function route() {
  const requested=location.hash.slice(1) || 'overview';
  const page=requested==='learning'?'followups':requested;
  if(requested==='learning')history.replaceState(null,'','#followups');
  state.page=['overview','inbox','capture','followups','catalog','connections','pesashield'].includes(page)?page:'overview';
  document.querySelectorAll('.page').forEach(element=>{element.hidden=element.id!=='page-'+state.page;});
  document.querySelectorAll('[data-nav]').forEach(link=>{
    const selected=link.dataset.nav===state.page;link.classList.toggle('active',selected);
    if(selected)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current');
  });
  $('page-label').textContent={overview:'Opportunities',inbox:'Signal inbox',capture:'Capture request',followups:'Follow-up queue',catalog:'Product aliases',connections:'Channels & import',pesashield:'PesaShield AI'}[state.page];
  window.scrollTo(0,0);
  if(state.page==='inbox')loadSignals().catch(showError);
  if(state.page==='catalog')loadCatalog().catch(showError);
  if(state.page==='followups')loadFollowups();
}
window.addEventListener('hashchange',route);
function showError(error) {$('global-error').textContent=error.message;$('global-error').hidden=false;}
async function refresh() {
  const revision=++state.requestVersion;
  try {
    const result=await api('/api/insights?'+query());
    if(revision!==state.requestVersion)return;
    state.insights=result;result.recent.forEach(row=>state.signals.set(row.id,row));
    renderOverview(result);$('global-error').hidden=true;
    $('workspace-status').textContent='Workspace ready';$('workspace-status').classList.remove('warn');
    if(state.page==='inbox')await loadSignals();
  } catch(error) {if(revision===state.requestVersion){showError(error);$('workspace-status').textContent='Connection unavailable';$('workspace-status').classList.add('warn');}}
}
function renderOverview(data) {
  const k=data.kpis;
  $('metrics').innerHTML=`<div class="metric"><span>Captured requests</span><b>${number(k.signals)}</b><small>Across channels in this view</small></div><div class="metric"><span>Active opportunities</span><b>${number(k.active_opportunities)}</b><small>Products with open requests</small></div><div class="metric"><span>Awaiting review</span><b>${number(k.needs_review)}</b><small>Check request details</small></div><div class="metric"><span>Recorded recoveries</span><b>${number(k.recovered)}</b><small>${esc(money(k.recovered_value)||'Record outcomes after follow-up')}</small></div>`;
  $('opportunity-count').textContent=`${data.demand.length} ${data.demand.length===1?'product':'products'}`;
  $('opportunities').innerHTML=data.demand.length ? data.demand.map(renderOpportunity).join('') : `<div class="empty-card"><h3>Your next opportunity starts here.</h3><p>${k.signals?'Try a broader filter to see customer demand.':'Capture a missed request to find patterns across products, channels and blockers.'}</p><a class="button primary" href="#capture">Capture the first request</a>${cfg.demo?' <button class="button" data-seed>Load retail demo</button>':' <a class="button" href="/demo">Explore sample demand</a>'}</div>`;
  const reasons=Object.entries(data.reasons).sort((a,b)=>b[1]-a[1]);
  const maximum=Math.max(1,...reasons.map(r=>r[1]));
  $('reason-bars').innerHTML=reasons.length?reasons.map(([reason,count])=>`<div class="bar-item"><div class="bar-label"><span>${esc(cfg.reasons[reason]||reason)}</span><b>${count}</b></div><div class="bar-track" role="img" aria-label="${esc(cfg.reasons[reason])}: ${count} requests"><div class="bar-fill" style="width:${100*count/maximum}%"></div></div></div>`).join(''):'<p class="footnote">Blockers appear once requests are captured.</p>';
  $('source-list').innerHTML=Object.entries(data.sources).sort((a,b)=>b[1]-a[1]).map(([source,count])=>`<div class="source-item"><span class="source-name"><i class="source-dot" aria-hidden="true"></i>${esc(cfg.sources[source]||source)}</span><b>${count}</b></div>`).join('')||'<p class="footnote">No channel activity in this view.</p>';
  $('budget-note').textContent=k.signals?`Open requests include ${money(k.stated_budgets)||'no stated budgets'}. These are expressions of interest, not lost revenue.`:'Customer budgets stay separated by currency. Record a recovery only when your team confirms an outcome.';
}
function renderOpportunity(item) {
  const tag=item.state==='resolved'?'green':item.priority==='High'?'amber':'';
  const initials=item.product.split(/\s+/).slice(0,2).map(word=>word[0]||'').join('').toUpperCase();
  return `<article class="opportunity-card"><div class="opportunity-top"><span class="product-avatar" aria-hidden="true">${esc(initials)}</span><span class="tag ${tag}">${esc(item.state==='resolved'?'Resolved':item.priority+' priority')}</span></div><h3>${esc(item.product)}</h3><div class="evidence-line"><b>${item.count}</b> requests · <b>${item.customers}</b> known customers · <b>${Object.keys(item.sources).length}</b> channels</div><div class="budget-line">${esc(money(item.budgets)?'Stated budgets: '+money(item.budgets):'No budget stated on open requests')}${item.urgent?` · ${item.urgent} urgent`:''}</div><div class="opportunity-action"><span>${item.state==='resolved'?'OUTCOME RECORDED':'SUGGESTED NEXT STEP'}</span><b>${esc(item.state==='resolved'?'All requests closed or recovered':item.suggested_action)}</b><p>${esc(item.action_basis)}</p></div><div class="opportunity-footer"><small>${item.open_requests} open · ${item.review_count} to review</small><button data-evidence="${esc(item.key)}">View evidence ↗</button></div></article>`;
}
async function loadSignals() {
  const revision=++state.signalRequestVersion;
  const data=await api('/api/signals?'+query({page:state.signalPage}));
  if(state.page!=='inbox'||revision!==state.signalRequestVersion)return;
  data.items.forEach(row=>state.signals.set(row.id,row));
  $('signal-count').textContent=`${number(data.total)} requests`;
  $('signal-rows').innerHTML=data.items.length?data.items.map(row=>`<tr><td><b>${esc(row.product)}</b><small>${esc(row.customer||'Customer not identified')} · ${date(row.ts)}</small></td><td>${esc(cfg.sources[row.source])}</td><td><span class="tag">${esc(cfg.reasons[row.effective_reason])}</span><small>${row.needs_review?'○ Needs review':'✓ Human reviewed'}</small></td><td><span class="tag ${row.status==='recovered'?'green':row.status==='follow_up'?'blue':''}">${esc(cfg.statuses[row.status])}</span><small>${esc(row.owner||'No owner yet')}</small></td><td><button class="button small" data-signal="${row.id}">Review →</button></td></tr>`).join(''):'<tr><td colspan="5">No requests match this view. Try a broader filter or capture a request.</td></tr>';
  $('page-number').textContent=`Page ${data.page} of ${data.pages}`;
  $('previous-page').disabled=data.page<=1;$('next-page').disabled=data.page>=data.pages;
}
$('previous-page').onclick=()=>{state.signalPage--;loadSignals().catch(showError);};
$('next-page').onclick=()=>{state.signalPage++;loadSignals().catch(showError);};
async function openEvidence(key) {
  const data=await api('/api/opportunities/evidence?'+query({key}));
  $('evidence-title').textContent=data.product;
  data.items.forEach(row=>state.signals.set(row.id,row));
  const group=data.summary;
  $('evidence-content').innerHTML=`<div class="evidence-summary"><b>${esc(group.suggested_action)}</b><p>${esc(group.action_detail)}</p><p class="footnote">${esc(group.action_basis)} ${esc(group.trend)} ${group.unidentified_requests} requests have no customer reference; these are not counted as distinct buyers.</p></div>`+data.items.map(row=>`<article class="evidence-request"><div class="section-heading"><b>${esc(cfg.sources[row.source])} · ${esc(row.customer||'Customer not identified')}</b><span class="tag ${row.status==='recovered'?'green':''}">${esc(cfg.statuses[row.status])}</span></div><blockquote class="quote">${esc(row.raw)}</blockquote><div class="signal-meta">${date(row.ts)} · ${esc(cfg.reasons[row.effective_reason])} · ${row.needs_review?'Needs human review':'Human reviewed'}${row.budget!==null?' · Stated budget '+esc(row.currency)+' '+number(row.budget):''}</div><button class="button small" data-signal="${row.id}">Review & choose a next step →</button></article>`).join('');
  $('evidence-dialog').showModal();
}
function openSignal(id) {
  const row=state.signals.get(Number(id));if(!row)return;
  if($('evidence-dialog').open)$('evidence-dialog').close();
  $('signal-evidence').innerHTML=`<div class="signal-meta">${esc(cfg.sources[row.source])} · ${esc(row.customer||'Customer not identified')} · ${date(row.ts)}</div><blockquote class="quote">${esc(row.raw)}</blockquote><p class="signal-meta">${row.needs_review?'Details awaiting review':'Details reviewed'} · ${esc(cfg.reasons[row.effective_reason])}</p>`;
  $('edit-id').value=row.id;$('edit-product').value=row.product;$('edit-qty').value=row.qty;
  $('edit-budget').value=row.budget??'';$('edit-currency').value=row.currency;
  $('edit-reason').value=row.effective_reason;$('edit-urgency').value=row.urgency;
  $('edit-status').value=row.status;$('edit-follow-up-on').value=row.follow_up_on||'';$('edit-owner').value=row.owner||'';$('edit-action').value=row.next_action||'';
  $('edit-recovered').value=row.recovered_amount??'';$('edit-recovered-currency').value=row.recovered_currency||row.currency||cfg.defaultCurrency;
  $('recovery-fields').hidden=row.status!=='recovered';$('review-error').textContent='';
  $('signal-dialog').showModal();
}
$('edit-status').onchange=()=>{$('recovery-fields').hidden=$('edit-status').value!=='recovered';};
$('signal-form').onsubmit=async event=>{
  event.preventDefault();
  await busy($('save-review'),'Saving…',async()=>{
    try {
      await api('/api/signals/'+$('edit-id').value,{method:'PATCH',body:JSON.stringify({
        product:$('edit-product').value.trim(),qty:Number($('edit-qty').value),budget:$('edit-budget').value===''?null:Number($('edit-budget').value),
        currency:$('edit-currency').value,reason:$('edit-reason').value,urgency:Number($('edit-urgency').value),status:$('edit-status').value,
        owner:$('edit-owner').value,next_action:$('edit-action').value,follow_up_on:$('edit-follow-up-on').value||null,
        recovered_amount:$('edit-recovered').value===''?null:Number($('edit-recovered').value),recovered_currency:$('edit-recovered-currency').value
      })});
      $('signal-dialog').close();toast('Request details and next step saved.');
      await Promise.all([refresh(),loadFollowups()]);
    }catch(error){$('review-error').textContent=error.message;}
  });
};
document.addEventListener('click',event=>{
  const button=event.target.closest('button');if(!button)return;
  if(button.dataset.close)$(button.dataset.close).close();
  if(button.dataset.evidence)openEvidence(button.dataset.evidence).catch(error=>toast(error.message,true));
  if(button.dataset.signal)openSignal(button.dataset.signal);
  if(button.dataset.followup)openFollowup(button.dataset.followup);
  if(button.hasAttribute('data-seed'))seedDemo(button);
});
function invalidateDraft(){state.captureVersion++;state.draft=null;$('draft-form').hidden=true;$('draft-empty').hidden=false;}
['capture-text','capture-source','capture-customer','capture-currency'].forEach(id=>$(id).addEventListener('input',invalidateDraft));
const examples={stock:{source:'whatsapp',text:'Do you have 2 wireless earbuds? Your website says sold out. Budget KES 6,000; I need them today.'},price:{source:'rejected_quote',text:'The customer declined the quote for 4 USB-C fast chargers because the price was too high. Total budget KES 4,000.'},delivery:{source:'chatbot',text:'Need an everyday backpack by tomorrow, but your delivery takes a week. Budget KES 4,200.'}};
document.querySelectorAll('[data-example]').forEach(button=>button.onclick=()=>{
  const example=examples[button.dataset.example];$('capture-text').value=example.text;$('capture-source').value=example.source;invalidateDraft();$('capture-text').focus();
});
$('capture-form').onsubmit=async event=>{
  event.preventDefault();
  await busy($('preview-button'),'Finding the signal…',async()=>{
    try {
      const payload={text:$('capture-text').value.trim(),source:$('capture-source').value,customer:$('capture-customer').value.trim()||null,
        currency:$('capture-currency').value,external_id:'manual-'+crypto.randomUUID()};
      const revision=state.captureVersion;
      const preview=await api('/api/signals/preview',{method:'POST',body:JSON.stringify(payload)});
      if(revision!==state.captureVersion){toast('The request changed while processing. Preview the updated message again.');return;}
      state.draft=preview;
      const draft=state.draft.fields;
      $('draft-product').value=draft.product;$('draft-qty').value=draft.qty;$('draft-budget').value=draft.budget??'';
      $('draft-currency').value=draft.currency;$('draft-reason').value=draft.reason;$('draft-urgency').value=draft.urgency;
      $('draft-location').value=draft.location||'';$('draft-verified').checked=false;
      $('draft-engine').textContent='Suggested request details. Check the product, quantity, budget and blocker before saving.';
      $('draft-warnings').innerHTML=state.draft.warnings.map(warning=>`<li>${esc(warning)}</li>`).join('');
      $('draft-empty').hidden=true;$('draft-form').hidden=false;$('capture-success').hidden=true;
      $('draft-product').focus();
    }catch(error){toast(error.message,true);}
  });
};
$('draft-form').onsubmit=async event=>{
  event.preventDefault();if(!state.draft)return;
  await busy($('save-button'),'Saving request…',async()=>{
    try {
      const {fields,warnings,...payload}=state.draft;
      payload.fields={product:$('draft-product').value.trim(),qty:Number($('draft-qty').value),budget:$('draft-budget').value===''?null:Number($('draft-budget').value),
        currency:$('draft-currency').value,reason:$('draft-reason').value,urgency:Number($('draft-urgency').value),location:$('draft-location').value.trim()||null};
      payload.verified=$('draft-verified').checked;
      const saved=await api('/api/signals',{method:'POST',body:JSON.stringify(payload)});
      $('capture-success').innerHTML=`<b>${saved.deduplicated?'This request was already saved.':'Request captured.'}</b> ${esc(saved.product)} now contributes to your opportunity board. <a class="text-link" href="#overview">See the opportunity →</a>`;
      $('capture-success').hidden=false;if($('capture-text').value.trim()===payload.text)$('capture-text').value='';invalidateDraft();toast('Request saved. The original conversation is kept as evidence.');
      await Promise.all([refresh(),loadFollowups()]);
    }catch(error){toast(error.message,true);}
  });
};
const SpeechRecognition=window.SpeechRecognition||window.webkitSpeechRecognition;
if(!SpeechRecognition){$('dictate').disabled=true;$('dictate').textContent='Voice unavailable in this browser';}
else {
  let listening=false;
  $('dictate').onclick=()=>{
    if(listening){recognition.stop();return;}
    recognition=new SpeechRecognition();recognition.lang=navigator.language||'en-US';recognition.interimResults=false;recognition.continuous=false;
    recognition.onstart=()=>{listening=true;$('dictate').textContent='■ Stop listening';$('preview-button').disabled=true;toast('Listening. Your transcript will appear for review before saving.');};
    recognition.onresult=event=>{const transcript=Array.from(event.results).map(result=>result[0].transcript).join(' ');$('capture-text').value=($('capture-text').value+' '+transcript).trim().slice(0,10000);$('capture-source').value='salesperson_voice';invalidateDraft();};
    recognition.onerror=event=>toast(event.error==='not-allowed'?'Microphone access was not granted. Type or paste the note instead.':'Voice transcription was unavailable. Type or paste the note instead.',true);
    recognition.onend=()=>{listening=false;$('dictate').textContent='♩ Dictate a note';$('preview-button').disabled=false;};
    try{recognition.start();}catch{toast('Voice transcription could not start. Type or paste the note instead.',true);}
  };
}
async function loadFollowups() {
  const revision=++state.followupsVersion;
  const filters=state.followups;
  const params=new URLSearchParams({status:filters.status,q:filters.q,urgency:filters.urgency,due:filters.due,page:filters.page});
  if(filters.owner==='unassigned')params.set('assignment','unassigned');
  else if(filters.owner.startsWith('owner:'))params.set('owner',filters.owner.slice(6));
  try {
    const data=await api('/api/operations/follow-ups?'+params);
    if(revision!==state.followupsVersion)return;
    data.items.forEach(row=>state.signals.set(row.id,row));state.followups.page=data.page;
    const summary=data.summary;
    $('nav-review').textContent=summary.needs_review||'';
    $('followups-metrics').innerHTML=`<div class="metric"><span>Active requests</span><b>${summary.active}</b><small>${summary.following_up} already in progress</small></div><div class="metric"><span>Overdue follow-ups</span><b>${summary.overdue}</b><small>Open work past its follow-up date</small></div><div class="metric"><span>Need an owner</span><b>${summary.unassigned}</b><small>Assign someone to take the next step</small></div><div class="metric"><span>Recorded recoveries</span><b>${summary.recovered}</b><small>${esc(money(summary.recovered_value)||'Record confirmed outcomes')}</small></div>`;
    const owners=[...data.owners];
    if(filters.owner.startsWith('owner:')&&!owners.includes(filters.owner.slice(6)))owners.push(filters.owner.slice(6));
    const options='<option value="">All owners</option><option value="unassigned">Unassigned</option>'+owners.map(owner=>`<option value="${esc('owner:'+owner)}">${esc(owner)}</option>`).join('');
    if($('followups-owner').innerHTML!==options)$('followups-owner').innerHTML=options;
    $('followups-owner').value=filters.owner;
    $('followups-count').textContent=`${data.total} ${data.total===1?'request':'requests'} in this view`;
    $('followups-list').innerHTML=data.items.length?data.items.map(renderFollowup).join(''):`<div class="empty-card"><h3>${summary.active?'No requests match this view.':'Your follow-up queue is clear.'}</h3><p>${summary.active?'Try another owner, date or status filter to find the work you need.':'Capture a customer request, then assign an owner and next step here.'}</p><a class="button primary" href="#capture">Capture a request</a></div>`;
    $('followups-page').textContent=`Page ${data.page} of ${data.pages}`;
    $('followups-previous').disabled=data.page<=1;$('followups-next').disabled=data.page>=data.pages;
    $('followups-error').hidden=true;
  }catch(error){
    if(revision!==state.followupsVersion)return;
    $('followups-error').textContent=error.message;$('followups-error').hidden=false;
  }
}
function followupDate(value){return new Date(value+'T12:00:00').toLocaleDateString(undefined,{month:'short',day:'numeric',year:'numeric'});}
function renderFollowup(row){
  const completed=['recovered','closed'].includes(row.status);
  const dueLabel=completed?'Completed':row.overdue?'Overdue · '+followupDate(row.follow_up_on):row.due_today?'Due today':row.follow_up_on?'Due '+followupDate(row.follow_up_on):'Not scheduled';
  const action=row.next_action||row.suggested_action;
  return `<article class="card followup-card"><div class="section-heading"><span class="tag ${row.status==='recovered'?'green':row.status==='follow_up'?'blue':''}">${esc(cfg.statuses[row.status])}</span><span class="tag ${row.overdue?'red':row.due_today?'amber':''}">${esc(dueLabel)}</span></div><h2>${esc(row.product)}</h2><p class="followup-customer">${esc(row.customer||'Customer reference not recorded')} · ${esc(cfg.sources[row.source])}</p><div class="followup-blocker"><span class="tag">${esc(cfg.reasons[row.effective_reason])}</span>${row.urgency===3&&!completed?'<span class="tag amber">Urgent</span>':''}</div><div class="followup-next"><span>${row.next_action?'NEXT STEP / OUTCOME':'SUGGESTED NEXT STEP'}</span><p>${esc(action)}</p></div><div class="followup-card-footer"><span><b>${esc(row.owner||'Unassigned')}</b><small>${row.status==='recovered'&&row.recovered_amount!==null?'Recorded recovery: '+esc(row.recovered_currency||'UNKNOWN')+' '+number(row.recovered_amount):'Captured '+date(row.ts)}</small></span><button class="button small" data-followup="${row.id}">${completed?'View outcome':'Manage follow-up'} →</button></div></article>`;
}
let followupSearchTimer;
$('followups-search').oninput=event=>{
  state.followups.q=event.target.value;state.followups.page=1;
  clearTimeout(followupSearchTimer);followupSearchTimer=setTimeout(loadFollowups,250);
};
for(const [id,key] of [['followups-owner','owner'],['followups-urgency','urgency'],['followups-due','due']]){
  $(id).onchange=event=>{state.followups[key]=event.target.value;state.followups.page=1;loadFollowups();};
}
document.querySelectorAll('[data-followup-status]').forEach(button=>button.onclick=()=>{
  state.followups.status=button.dataset.followupStatus;state.followups.page=1;
  const completed=['recovered','closed'].includes(state.followups.status);
  if(completed){state.followups.due='all';$('followups-due').value='all';}
  $('followups-due').disabled=completed;
  document.querySelectorAll('[data-followup-status]').forEach(tab=>{const active=tab===button;tab.classList.toggle('active',active);tab.setAttribute('aria-pressed',String(active));});
  loadFollowups();
});
$('followups-previous').onclick=()=>{state.followups.page--;loadFollowups();};
$('followups-next').onclick=()=>{state.followups.page++;loadFollowups();};
function openFollowup(id){
  const row=state.signals.get(Number(id));if(!row)return;
  $('followup-context').innerHTML=`<h3>${esc(row.product)}</h3><p class="signal-meta">${esc(row.customer||'Customer reference not recorded')} · ${esc(cfg.sources[row.source])} · ${esc(cfg.reasons[row.effective_reason])}</p><blockquote class="quote">${esc(row.raw)}</blockquote>`;
  $('followup-id').value=row.id;$('followup-owner').value=row.owner||'';$('followup-status').value=row.status;
  $('followup-date').value=row.follow_up_on||'';$('followup-action').value=row.next_action||'';
  $('followup-recovered').value=row.recovered_amount??'';$('followup-recovery-currency').value=row.recovered_currency||row.currency||cfg.defaultCurrency;
  $('followup-recovery-fields').hidden=row.status!=='recovered';$('followup-save-error').textContent='';
  $('followup-dialog').showModal();
}
$('followup-status').onchange=()=>{$('followup-recovery-fields').hidden=$('followup-status').value!=='recovered';};
$('followup-form').onsubmit=async event=>{
  event.preventDefault();
  await busy($('save-followup'),'Saving…',async()=>{
    try{
      await api('/api/signals/'+$('followup-id').value,{method:'PATCH',body:JSON.stringify({
        owner:$('followup-owner').value,status:$('followup-status').value,follow_up_on:$('followup-date').value||null,
        next_action:$('followup-action').value,recovered_amount:$('followup-recovered').value===''?null:Number($('followup-recovered').value),
        recovered_currency:$('followup-recovery-currency').value
      })});
      $('followup-dialog').close();toast('Follow-up saved. Your team’s queue and outcome totals are updated.');
      await Promise.all([refresh(),loadFollowups()]);
    }catch(error){$('followup-save-error').textContent=error.message;}
  });
};
async function loadCatalog(){
  const result=await api('/api/catalog');$('catalog-count').textContent=result.items.length+' products';
  $('catalog-list').innerHTML=result.items.length?result.items.map(item=>`<div class="catalog-item"><b>${esc(item.name)}</b><small>${esc(item.sku||'No SKU')}</small><div class="alias-tags">${item.aliases.map(alias=>`<span class="tag">${esc(alias)}</span>`).join('')}</div></div>`).join(''):'<p class="muted">No mappings yet. Add the names your customers use to connect related requests.</p>';
}
$('catalog-form').onsubmit=async event=>{
  event.preventDefault();const button=event.target.querySelector('button[type=submit]');
  await busy(button,'Saving mapping…',async()=>{
    try{await api('/api/catalog',{method:'POST',body:JSON.stringify({name:$('catalog-name').value.trim(),sku:$('catalog-sku').value.trim(),aliases:$('catalog-aliases').value.split('\n').map(s=>s.trim()).filter(Boolean)})});event.target.reset();await Promise.all([loadCatalog(),refresh()]);toast('Product mapping saved. Matching product names are now grouped together.');}
    catch(error){toast(error.message,true);}
  });
};
async function seedDemo(button){
  await busy(button,'Loading sample demand…',async()=>{
    try{const result=await api('/api/demo/seed',{method:'POST'});toast(result.message);await Promise.all([refresh(),loadFollowups()]);}
    catch(error){toast(error.message,true);}
  });
}
if($('seed-demo'))$('seed-demo').onclick=()=>seedDemo($('seed-demo'));
$('import-form').onsubmit=async event=>{
  event.preventDefault();await busy($('preview-import'),'Checking CSV…',async()=>{
    try{
      const form=new FormData();form.append('file',$('import-file').files[0]);const result=await api('/api/import/preview',{method:'POST',body:form});state.imports=result.drafts;
      $('import-preview').hidden=false;$('import-preview').innerHTML=`<div class="import-results"><h3>${result.drafts.length} ready · ${result.errors.length} need fixing</h3><p>${result.total} rows checked. Preview of extracted requests:</p><ul>${result.drafts.slice(0,5).map(d=>`<li><b>${esc(d.fields.product)}</b> · ${esc(cfg.sources[d.source])} · ${esc(cfg.reasons[d.fields.reason])}</li>`).join('')}</ul>${result.errors.length?`<p class="error-text">Rows below will not be imported:</p><ul>${result.errors.map(error=>`<li>Row ${error.row}: ${esc(error.error)}</li>`).join('')}</ul>`:''}${result.drafts.length?`<button class="button primary" id="confirm-import">Import ${result.drafts.length} valid requests</button>`:''}</div>`;
      if($('confirm-import'))$('confirm-import').onclick=confirmImport;
    }catch(error){toast(error.message,true);}
  });
};
async function confirmImport(){
  await busy($('confirm-import'),'Importing…',async()=>{
    try{
      const items=state.imports.map(({fields,warnings,...item})=>item);
      const results=await api('/api/signals/bulk',{method:'POST',body:JSON.stringify({items})});
      const duplicate=results.filter(item=>item.deduplicated).length;
      $('import-preview').innerHTML=`<div class="success-panel">${results.length-duplicate} requests imported. ${duplicate} existing events were counted once. <a class="text-link" href="#inbox">Review the inbox →</a></div>`;
      state.imports=[];await Promise.all([refresh(),loadFollowups()]);toast('Import complete. Review the captured request details in the signal inbox.');
    }catch(error){toast(error.message,true);}
  });
}
function download(blob,name){const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=name;document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('csv-template').onclick=()=>download(new Blob(['text,source,customer,currency,external_id\n"Need 2 wireless earbuds but they are sold out",whatsapp,Shopper A,KES,example-1\n"usb c fast charger",website_search,,KES,example-2\n'],{type:'text/csv'}),'ghostdemand-import-example.csv');
$('export').onclick=()=>busy($('export'),'Preparing CSV…',async()=>{
  try{const response=await fetch('/api/signals/export?'+query(),{headers:cfg.demo?{'X-Ghost-Demo':'1'}:{}});if(!response.ok)throw new Error('Unable to export this view.');download(await response.blob(),'ghostdemand-signals.csv');}
  catch(error){toast(error.message,true);}
});
route();refresh();if(state.page!=='followups')loadFollowups();


async function loadFraudInsights(){
  try{
    const res = await fetch(api('/api/pesashield/insights'));
    if(!res.ok) return;
    const data = await res.json();
    if($('fraud-total')) $('fraud-total').textContent = data.kpis.total;
    if($('fraud-scam')) $('fraud-scam').textContent = data.kpis.scam;
    if($('fraud-suspicious')) $('fraud-suspicious').textContent = data.kpis.suspicious;
    if($('fraud-reviewed')) $('fraud-reviewed').textContent = data.kpis.reviewed;
  }catch(e){}
}

async function loadFraudReports(){
  try{
    const res = await fetch(api('/api/pesashield/reports'));
    if(!res.ok) return;
    const data = await res.json();
    const rows = $('fraud-rows');
    if(!rows) return;
    if(!data.items.length){rows.innerHTML='<tr><td colspan="6">No reports yet.</td></tr>';return;}
    rows.innerHTML = data.items.map(r=>`<tr><td><small>${esc(r.ts?.slice(0,16)||'')}</small></td><td><span class="tag ${r.verdict==='scam'?'amber':r.verdict==='suspicious'?'blue':''}">${esc(r.verdict)}</span>${r.reviewed_verdict?'<small>✓ '+esc(r.reviewed_verdict)+'</small>':''}</td><td>${r.risk_score?.toFixed? r.risk_score.toFixed(2):r.risk_score}</td><td><small>${(r.reason_codes||[]).join(', ')}</small></td><td><small>${esc(r.body_preview)}</small></td><td><button class="button small" data-fraud-review="${r.id}" data-verdict="scam">Mark scam</button> <button class="button small" data-fraud-review="${r.id}" data-verdict="legit">Mark legit</button></td></tr>`).join('');
    rows.querySelectorAll('[data-fraud-review]').forEach(btn=>{
      btn.onclick=async()=>{
        const id=btn.dataset.fraudReview, verdict=btn.dataset.verdict;
        const res=await fetch(api('/api/pesashield/reports/'+id+'/review'),{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({verdict})});
        if(res.ok){toast('Review saved');loadFraudReports();loadFraudInsights();}
        else{const j=await res.json().catch(()=>({}));toast(j.error||'Failed');}
      };
    });
  }catch(e){}
}

async function loadFraudModel(){
  try{
    const res = await fetch(api('/api/pesashield/model'));
    if(!res.ok) return;
    const data = await res.json();
    const el = $('fraud-model-status');
    if(!el) return;
    el.innerHTML = `<small>Ready: ${data.ready?'yes':'no'} · Threshold: ${data.threshold}<br>Version: ${data.metadata?.version||'-'}</small>`;
  }catch(e){}
}


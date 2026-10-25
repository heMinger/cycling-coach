const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('index.html','utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
function el(id) {
  if (!elements.has(id)) elements.set(id,{value:'',innerHTML:'',textContent:'',disabled:false,style:{},
    classList:{add(){},remove(){},toggle(){}},checkValidity:()=>true, appendChild(){},remove(){},focus(){}});
  return elements.get(id);
}
const ctx = vm.createContext({location:{protocol:'http:',origin:'http://localhost:8000'},
  document:{getElementById:el,createElement:()=>el('message-'+Math.random()),querySelectorAll:()=>[]},
  localStorage:{getItem:()=> 'test-session',setItem(){}},crypto:{randomUUID:()=> 'test-session'},
  setTimeout:()=>{}, console});
vm.runInContext(script,ctx);
const event = {date:'2026-10-26',name:'耐力课',kind:'ride',description:'热身\n耐力',duration_minutes:60,evidence_ids:['metrics']};
ctx.result={plan:{status:'draft',summary:'<img src=x onerror=alert(1)>',assumptions:[],questions:[],events:[event],total_minutes:60},
  constraints:{start_date:'2026-10-26',days:7},evidence:[{id:'metrics',text:'研究与局限'}]};
vm.runInContext('renderPlan(result)',ctx);
assert.match(el('plan-content').innerHTML,/耐力课/);
assert.match(el('plan-content').innerHTML,/&lt;img/);
assert.ok(!el('plan-content').innerHTML.includes('<img'));
assert.match(el('plan-content').innerHTML,/研究与局限/);
console.log('PASS nested API response renders, evidence visible, model text escaped');
ctx.result.plan={status:'needs_info',summary:'请补充',assumptions:[],questions:['最近练了多少'],events:[]};
vm.runInContext('renderPlan(result)',ctx);
assert.match(el('plan-content').innerHTML,/最近练了多少/);
assert.ok(!el('plan-content').innerHTML.includes('writeToPlan'));
console.log('PASS missing-data response shows questions without calendar write action');
(async()=>{
  let sent;
  el('plan-input').value='下周计划'; el('plan-start').value='2026-10-26';
  el('plan-hours').value='3'; el('plan-daily').value='60'; el('plan-ftp').value='200';
  el('plan-offdays').value='2026-10-28';
  ctx.fetch=async(url,opts)=>{sent=JSON.parse(opts.body); return {ok:true,json:async()=>ctx.result};};
  await vm.runInContext('generatePlan()',ctx);
  assert.equal(sent.constraints.max_minutes,180);
  assert.equal(sent.constraints.max_daily_minutes,60);
  assert.equal(sent.constraints.ftp,200);
  assert.equal(sent.constraints.unavailable_dates[0],'2026-10-28');
  assert.equal(sent.session_id,'test-session');
  console.log('PASS generation sends explicit time, date, FTP and session constraints');
  ctx.result.plan={status:'draft',summary:'test',assumptions:[],questions:[],events:[event],total_minutes:60};
  vm.runInContext('renderPlan(result)',ctx);
  ctx.fetch=async(url,opts)=>{sent=JSON.parse(opts.body); return {ok:true,json:async()=>({created:1})};};
  await vm.runInContext('writeToPlan()',ctx);
  assert.equal(sent.plan.events[0].kind,'ride');
  assert.ok(sent.constraints);
  assert.equal(el('write-btn').disabled,true);
  console.log('PASS write forwards canonical plan envelope; disables repeat click after success');
})().catch(e=>{console.error(e);process.exitCode=1;});

// Real browser + synthetic API responses. Never reaches model/calendar services.
const { chromium } = require('playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({channel:'chrome',headless:true});
  try {
    const page = await browser.newPage({viewport:{width:1200,height:900}});
    const errors=[]; page.on('pageerror',error=>errors.push(error.message));
    let writes=0, lastRequest, mode='draft';
    const result={constraints:{start_date:'2026-10-26',days:7,max_minutes:180},
      plan:{status:'draft',summary:'按近期训练安排耐力课，并记录完成感受。',assumptions:['当天无疼痛'],questions:[],total_minutes:60,
        events:[{date:'2026-10-26',kind:'ride',name:'耐力骑',purpose:'维持耐力',adjustment:'明显吃力则缩短',
          evidence_ids:['training_distribution'],steps:[{name:'耐力',minutes:60,target_pct_ftp:65}],
          description:'目的：维持耐力\n耐力：60 分钟 @ 65% FTP（约 130 W）\n调整条件：明显吃力则缩短',duration_minutes:60}]},
      evidence:[{id:'training_distribution',text:'研究结论与适用局限：不能外推为人人最优。'}]};
    await page.route('**/*',async route=>{
      const url = new URL(route.request().url());
      if(url.pathname==='/ui') return route.fulfill({contentType:'text/html',body:fs.readFileSync('index.html','utf8')});
      lastRequest=route.request().postDataJSON();
      if(url.pathname==='/plan/create') {
        writes++; assert.equal(lastRequest.plan.events[0].kind,'ride');
        return route.fulfill({json:{created:1}});
      }
      if(url.pathname==='/plan/generate') {
        if(mode==='error')return route.fulfill({status:422,json:{detail:'超过本周期可用总时间'}});
        if(mode==='needs_info')return route.fulfill({json:{...result,plan:{status:'needs_info',summary:'请补充近期训练',assumptions:[],questions:['最近每周骑行多久？'],events:[]}}});
        return route.fulfill({json:result});
      }
      if(url.pathname==='/ask')return route.fulfill({json:{answer:'已生成草案，请查看计划。',intent:'plan',plan_result:result}});
      return route.abort();
    });
    await page.goto('http://127.0.0.1:8765/ui');
    assert.ok(!(await page.locator('body').innerText()).includes('已读取'));
    assert.ok(!(await page.locator('.header').innerText()).includes('202W'));
    await page.locator('.tab').nth(1).click();
    await page.locator('#plan-hours').fill('3');
    await page.locator('#plan-daily').fill('60');
    await page.locator('#plan-ftp').fill('200');
    await page.locator('#gen-btn').click();
    await page.locator('.event-card').waitFor();
    assert.equal(lastRequest.constraints.max_minutes,180);
    assert.ok((await page.locator('#plan-content').innerText()).includes('130 W'));
    await page.locator('details summary').click();
    assert.ok(await page.getByText('研究结论与适用局限：不能外推为人人最优。', {exact:true}).isVisible());
    await page.locator('#write-btn').click();
    await page.waitForFunction(()=>document.getElementById('write-btn').textContent==='已写入');
    assert.equal(writes,1); assert.ok(await page.locator('#write-btn').isDisabled());
    console.log('PASS real browser generates, renders evidence and sends validated-envelope calendar request to stub');
    mode='needs_info'; await page.locator('#gen-btn').click();
    await page.getByText('请补充：最近每周骑行多久？').waitFor();
    assert.equal(await page.locator('#write-btn').count(),0);
    console.log('PASS real browser needs-info state has no write button');
    mode='error'; await page.locator('#gen-btn').click();
    await page.getByText('生成失败：超过本周期可用总时间').waitFor();
    assert.ok(await page.locator('#gen-btn').isEnabled());
    console.log('PASS real browser server validation error is actionable and allows retry');
    mode='draft'; await page.locator('.tab').nth(0).click();
    await page.locator('#chat-input').fill('给我计划'); await page.locator('#send-btn').click();
    await page.getByText('已生成草案，请查看计划。').waitFor();
    await page.locator('.tab').nth(1).click(); await page.locator('.event-card').waitFor();
    console.log('PASS real browser chat result appears in plan tab without a second model call');
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
    await page.screenshot({path:'/private/tmp/cycling-coach-mobile.png',fullPage:true});
    assert.deepEqual(errors,[]);
    console.log('PASS mobile viewport has no horizontal overflow; no browser JavaScript errors');
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});

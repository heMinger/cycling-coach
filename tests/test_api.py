"""Exercise real FastAPI routes; replace external model/vector startup and calendar I/O."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from datetime import timedelta
from unittest.mock import patch
from test_planning import draft
from planning import generate_grounded_plan, today_local

HAS_DEPS = all(importlib.util.find_spec(n) for n in ('fastapi','httpx','dotenv','requests'))

@unittest.skipUnless(HAS_DEPS, 'install requirements-test.txt for API integration tests')
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake = types.ModuleType('rag')
        for name in ('load_documents','split_documents','build_vectorstore','build_chain','route_request','run_plan_pipeline'):
            setattr(fake,name,lambda *a,**k: None)
        with patch.dict(sys.modules, {'rag':fake}):
            import api
        cls.api=api
        from fastapi.testclient import TestClient
        cls.client=TestClient(api.app)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.db_patch=patch.object(self.api.mem,'DB_PATH',self.tmp.name+'/memory.db')
        self.db_patch.start(); self.api.mem.init_db()
        self.start=(today_local()+timedelta(days=1)).isoformat()
        self.c={'start_date':self.start,'max_minutes':180,'max_daily_minutes':60,'ftp':200}
        raw=draft(); raw['events'][0]['date']=self.start
        self.result=generate_grounded_plan('计划','测试数据',constraints=self.c,invoke=lambda _:json.dumps(raw))

    def tearDown(self):
        self.db_patch.stop(); self.tmp.cleanup()

    def test_generation_contract_and_history(self):
        self.api.mem.save_message('s1','user','周三没时间')
        with patch.object(self.api,'run_plan_pipeline',return_value=self.result) as run:
            r=self.client.post('/plan/generate',json={'question':'生成计划','session_id':'s1','constraints':self.c})
        self.assertEqual(r.status_code,200)
        self.assertIn('plan',r.json()); self.assertIn('constraints',r.json())
        self.assertIn('周三没时间',run.call_args.args[1])

    def test_bad_generation_is_422(self):
        with patch.object(self.api,'run_plan_pipeline',side_effect=ValueError('超过总时间')):
            r=self.client.post('/plan/generate',json={'question':'计划'})
        self.assertEqual(r.status_code,422)

    def test_chat_returns_actual_plan(self):
        with patch.object(self.api,'chain',object()), patch.object(self.api,'route_request',return_value='plan'), \
             patch.object(self.api,'run_plan_pipeline',return_value=self.result), \
             patch.object(self.api.mem,'process_memory_update'):
            r=self.client.post('/ask',json={'question':'生成计划','constraints':self.c})
        self.assertEqual(r.status_code,200)
        self.assertEqual(r.json()['plan_result']['plan']['events'][0]['date'],self.start)

    def test_write_revalidates_before_external_call(self):
        payload={'plan':self.result['plan'],'constraints':{**self.c,'max_minutes':10}}
        with patch('intervals_client.IntervalsClient') as client:
            r=self.client.post('/plan/create',json=payload)
        self.assertEqual(r.status_code,422); client.assert_not_called()

    def test_write_uses_canonical_description_and_omits_fake_tss(self):
        self.result['plan']['events'][0].update(description='WRONG',load_target=999)
        with patch('intervals_client.IntervalsClient') as client:
            client.return_value.create_event.return_value={'id':'test-only'}
            r=self.client.post('/plan/create',json={'plan':self.result['plan'],'constraints':self.c})
            kwargs=client.return_value.create_event.call_args.kwargs
        self.assertEqual(r.status_code,200)
        self.assertIsNone(kwargs['load_target']); self.assertNotIn('WRONG',kwargs['description'])

    def test_needs_info_cannot_write(self):
        p={'status':'needs_info','summary':'缺少数据','assumptions':[],'questions':['请刷新'],'events':[]}
        with patch('intervals_client.IntervalsClient') as client:
            r=self.client.post('/plan/create',json={'plan':p,'constraints':self.c})
        self.assertEqual(r.status_code,422); client.assert_not_called()

    def test_missing_load_is_not_zero_readiness(self):
        from intervals_client import IntervalsClient
        client=IntervalsClient(athlete_id='test',api_key='test')
        with patch.object(client,'get_athlete_info',return_value={}), \
             patch.object(client,'get_activities',return_value=[]), \
             patch.object(client,'get_events',return_value=[]), \
             patch.object(client,'get_wellness',return_value=[]):
            context=client.build_user_context()
        self.assertIn('状态值（TSB）：-',context)
        self.assertIn('HRV：未提供',context)

    def test_latest_wellness_sorted_and_zero_is_valid(self):
        from intervals_client import IntervalsClient
        client=IntervalsClient(athlete_id='test',api_key='test')
        wellness=[{'id':'2026-10-25','ctl':0,'atl':0,'hrv':50},
                  {'id':'2026-10-24','ctl':50,'atl':80}]
        with patch.object(client,'get_athlete_info',return_value={}), \
             patch.object(client,'get_activities',return_value=[]), \
             patch.object(client,'get_events',return_value=[]), \
             patch.object(client,'get_wellness',return_value=wellness):
            context=client.build_user_context()
        self.assertIn('数据日期：2026-10-25',context)
        self.assertIn('状态值（TSB）：0',context)
        self.assertIn('HRV：50',context)

    def test_zero_activity_days_does_not_duplicate_strava_activity(self):
        from intervals_client import IntervalsClient
        client=IntervalsClient(athlete_id='test',api_key='test')
        with patch.object(client,'get_athlete_info',return_value={}), \
             patch.object(client,'get_activities') as activities, \
             patch.object(client,'get_events',return_value=[]), \
             patch.object(client,'get_wellness',return_value=[]):
            context=client.build_user_context(activity_days=0)
        activities.assert_not_called()
        self.assertIn('不重复计入',context)

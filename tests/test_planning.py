import copy
import json
import unittest
from datetime import date
from planning import (generate_grounded_plan, normalize_constraints, validate_plan,
                      evidence_bundle, parse_object)

TODAY = date(2026, 10, 25)
CONSTRAINTS = {'start_date':'2026-10-26', 'max_minutes':180, 'max_daily_minutes':60, 'ftp':200}
IDS = {c['id'] for c in evidence_bundle()}

def draft():
    return {'status':'draft', 'summary':'根据已有稳定训练，安排舒适耐力，记录完成感受。',
            'assumptions':['当日没有不适，执行前确认'], 'questions':[],
            'events':[{'date':'2026-10-26', 'name':'耐力骑', 'kind':'ride',
                       'purpose':'在60分钟可用时间内维持耐力训练',
                       'adjustment':'明显吃力时缩短，疼痛时停止并确认情况',
                       'evidence_ids':['training_distribution'],
                       'steps':[{'name':'热身','minutes':10,'target_pct_ftp':50},
                                {'name':'耐力','minutes':40,'target_pct_ftp':65},
                                {'name':'冷身','minutes':10,'target_pct_ftp':45}]}]}

class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.c = normalize_constraints(CONSTRAINTS, TODAY)

    def test_canonical_duration_watts_and_untrusted_load(self):
        raw = draft()
        raw['events'][0].update(duration_minutes=5, load_target=999, description='错误的999W')
        p = validate_plan(raw,self.c,IDS)
        e = p['events'][0]
        self.assertEqual(e['duration_minutes'],60)
        self.assertIn('130 W',e['description'])
        self.assertNotIn('999',e['description'])
        self.assertIsNone(e['load_target'])
        self.assertEqual(e['estimated_load_proxy'],36)
        self.assertEqual(raw['events'][0]['duration_minutes'],5)

    def test_invalid_candidates(self):
        mutations = [
            ('past', lambda p:p['events'][0].update(date='2020-01-01')),
            ('outside', lambda p:p['events'][0].update(date='2026-11-20')),
            ('duplicate', lambda p:p['events'].append(copy.deepcopy(p['events'][0]))),
            ('daily_budget', lambda p:p['events'][0]['steps'][1].update(minutes=61)),
            ('unknown_source', lambda p:p['events'][0].update(evidence_ids=['made_up_paper'])),
            ('missing_steps', lambda p:p['events'][0].update(steps=[])),
            ('negative', lambda p:p['events'][0]['steps'][0].update(minutes=-1)),
            ('nan', lambda p:p['events'][0]['steps'][0].update(target_pct_ftp=float('nan'))),
            ('boolean', lambda p:p['events'][0]['steps'][0].update(minutes=True)),
            ('rest_with_steps',lambda p:p['events'][0].update(kind='rest')),
            ('missing_adjustment',lambda p:p['events'][0].update(adjustment='')),
        ]
        for name, change in mutations:
            with self.subTest(name=name):
                p=draft(); change(p)
                with self.assertRaises(ValueError):validate_plan(p,self.c,IDS)

    def test_total_budget(self):
        c={**self.c,'max_minutes':45}
        with self.assertRaisesRegex(ValueError,'总时间'): validate_plan(draft(),c,IDS)

    def test_unavailable_day(self):
        c={**self.c,'unavailable_dates':['2026-10-26']}
        with self.assertRaisesRegex(ValueError,'不可训练'): validate_plan(draft(),c,IDS)

    def test_rest_day(self):
        p=draft(); p['events'][0].update(kind='rest',steps=[])
        out=validate_plan(p,self.c,IDS)
        self.assertEqual(out['total_minutes'],0)
        self.assertEqual(out['events'][0]['load_target'],0)

    def test_missing_ftp_no_invented_watts(self):
        c={**self.c}; del c['ftp']
        self.assertNotIn(' W',validate_plan(draft(),c,IDS)['events'][0]['description'])

    def test_needs_info_no_executable_workouts(self):
        p={'status':'needs_info','summary':'请补充训练基础','assumptions':[],
           'questions':['近期每周训练多少？'],'events':[]}
        self.assertEqual(validate_plan(p,self.c,IDS)['status'],'needs_info')
        p['events']=draft()['events']
        with self.assertRaises(ValueError):validate_plan(p,self.c,IDS)

    def test_cached_data_makes_no_model_call(self):
        def fail(_): raise AssertionError('不应调用模型')
        for status in ('cached','unavailable'):
            result=generate_grounded_plan('下周计划',f'[DATA_STATUS:{status}]',
                invoke=fail,constraints=CONSTRAINTS,today=TODAY)
            self.assertEqual(result['plan']['events'],[])
            self.assertEqual(result['validation'],'needs_info')

    def test_prompt_contains_evidence_limits_history_and_current_window(self):
        prompts=[]
        def fake(prompt):prompts.append(prompt); return json.dumps(draft())
        result=generate_grounded_plan('耐力计划','近期每周3小时','周三没时间',
            history='上节课感觉很累',invoke=fake,constraints=CONSTRAINTS,today=TODAY)
        self.assertIn('上节课感觉很累',prompts[0])
        self.assertIn('34304689',prompts[0])
        self.assertIn('不能外推',prompts[0])
        self.assertIn('2026-10-26',prompts[0])
        self.assertEqual(result['validation'],'checked')

    def test_invalid_json_repair_once_then_fail_closed(self):
        calls=[]
        def fake(prompt):calls.append(prompt); return 'not json'
        with self.assertRaisesRegex(ValueError,'未通过校验'):
            generate_grounded_plan('计划','profile',invoke=fake,constraints=CONSTRAINTS,today=TODAY)
        self.assertEqual(len(calls),2)
        self.assertIn('未通过校验',calls[1])

    def test_over_budget_repaired(self):
        bad=draft(); bad['events'][0]['steps'][0]['minutes']=100
        outputs=iter([json.dumps(bad),json.dumps(draft())])
        result=generate_grounded_plan('计划','profile',invoke=lambda _:next(outputs),constraints=CONSTRAINTS,today=TODAY)
        self.assertEqual(result['plan']['total_minutes'],60)

    def test_constraints_validation(self):
        for c in [{'start_date':'2020-01-01'},{'ftp':False},{'days':2.5},{'max_minutes':-1}]:
            with self.subTest(c=c):
                with self.assertRaises(ValueError):normalize_constraints(c,TODAY)

    def test_json_object_only(self):
        self.assertEqual(parse_object('```json\n{"a":1}\n```'),{'a':1})
        with self.assertRaises(ValueError):parse_object('[]')

if __name__=='__main__':unittest.main()

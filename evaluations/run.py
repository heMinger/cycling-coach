"""Dry-run exports exact new prompts; --live calls DeepSeek using synthetic profiles only.
No Strava/Intervals access, no calendar writes. Live results need human review.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from datetime import timedelta
from urllib.request import Request, urlopen
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planning import (build_prompt,evidence_bundle,generate_grounded_plan,
                      normalize_constraints,today_local)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--live',action='store_true')
    parser.add_argument('--output',default='evaluations/latest.json')
    args=parser.parse_args()
    if args.live:
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parents[1]/'.env')
        except ImportError:
            pass
        if not os.getenv('DEEPSEEK_API_KEY'):
            parser.error('没有配置 DEEPSEEK_API_KEY；未发起模型请求。请在本地环境或 .env 配置。')
    today=today_local(); start=today+timedelta(days=1)
    cards=evidence_bundle(); rows=[]
    def invoke(prompt):
        body={'model':'deepseek-chat','temperature':0.2,
              'messages':[{'role':'system','content':prompt}]}
        req=Request('https://api.deepseek.com/chat/completions',
            data=json.dumps(body).encode(),headers={'Content-Type':'application/json',
            'Authorization':'Bearer '+os.environ['DEEPSEEK_API_KEY']})
        with urlopen(req,timeout=90) as r:
            data=json.load(r)
        return data['choices'][0]['message']['content']
    scenarios=json.loads(Path(__file__).with_name('scenarios.json').read_text())
    for case in scenarios:
        c={'start_date':start.isoformat(),'days':7}
        c.update({k:case[k] for k in ('max_minutes','max_daily_minutes','ftp') if k in case})
        if case['id']=='experienced_time_budget':
            c['unavailable_dates']=[(start+timedelta(days=i)).isoformat() for i in range(7) if (start+timedelta(days=i)).weekday()==2]
        c=normalize_constraints(c,today)
        row={'id':case['id'],'expected_status':case['expected_status'],'manual_review':case['review']}
        if args.live:
            try:
                result=generate_grounded_plan(case['request'],case['profile'],constraints=c,invoke=invoke,today=today)
                row.update(result=result,status_matches=result['plan']['status']==case['expected_status'])
            except Exception as exc:
                row['error_type']=type(exc).__name__
                row['error']='生成未完成；需要检查服务连接或模型输出。'
        else:
            row['prompt']=build_prompt(case['request'],case['profile'],'','',c,cards)
        rows.append(row)
    report={'mode':'live' if args.live else 'prompt_preview_only','date':today.isoformat(),
            'note':'状态匹配及程序校验不代表训练适用性已验证；按 manual_review 人工审核。','cases':rows}
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(f'{len(rows)} cases; mode={report["mode"]}; output={output}')

if __name__=='__main__':main()

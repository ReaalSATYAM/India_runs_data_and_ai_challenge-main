
import json
import collections
import statistics as st
import datetime

path = 'candidates.jsonl'

titles = collections.Counter()
countries = collections.Counter()
locs = collections.Counter()
inds = collections.Counter()

yoe = []
nskills = []
resp = []
gh = []
notice = []
last_active = []
salmax = []

honeypot_hits = 0
hp_examples = []

n = 0

def parse(d):
    try:
        return datetime.date.fromisoformat(d)
    except:
        return None

for line in open(path, encoding='utf-8'):
    line = line.strip()
    if not line:
        continue

    c = json.loads(line)
    n += 1

    p = c['profile']
    s = c['redrob_signals']

    titles[p['current_title']] += 1
    countries[p['country']] += 1
    locs[p['location']] += 1
    inds[p['current_industry']] += 1

    yoe.append(p['years_of_experience'])
    nskills.append(len(c['skills']))
    resp.append(s['recruiter_response_rate'])
    gh.append(s['github_activity_score'])
    notice.append(s['notice_period_days'])
    last_active.append(s['last_active_date'])
    salmax.append(s['expected_salary_range_inr_lpa']['max'])

    # Crude honeypot heuristic:
    # - expert skill with 0 months duration
    # - total career duration exceeds years of experience

    hp = False

    for sk in c['skills']:
        if sk.get('proficiency') == 'expert' and sk.get('duration_months', 0) == 0:
            hp = True

    tot = sum(r['duration_months'] for r in c['career_history'])

    if tot > p['years_of_experience'] * 12 + 18:
        hp = True

    if hp:
        honeypot_hits += 1
        if len(hp_examples) < 5:
            hp_examples.append(c['candidate_id'])

out = []

out.append('N=' + str(n))

out.append('--- TOP 60 TITLES ---')
for t, ct in titles.most_common(60):
    out.append(f'{ct:6d}  {t}')

out.append('--- TOP 15 COUNTRIES ---')
for t, ct in countries.most_common(15):
    out.append(f'{ct:6d}  {t}')

out.append('--- TOP 25 LOCATIONS ---')
for t, ct in locs.most_common(25):
    out.append(f'{ct:6d}  {t}')

out.append('--- TOP 20 INDUSTRIES ---')
for t, ct in inds.most_common(20):
    out.append(f'{ct:6d}  {t}')

def desc(name, a):
    a = [x for x in a if isinstance(x, (int, float))]
    a.sort()

    q = lambda p: a[int(p * (len(a) - 1))]

    out.append(
        f'{name}: '
        f'min={a[0]:.2f} '
        f'p10={q(.1):.2f} '
        f'p50={q(.5):.2f} '
        f'p90={q(.9):.2f} '
        f'max={a[-1]:.2f} '
        f'mean={sum(a)/len(a):.2f}'
    )

desc('years_of_experience', yoe)
desc('n_skills', nskills)
desc('recruiter_response_rate', resp)
desc('github_activity_score', gh)
desc('notice_period_days', notice)
desc('salary_max_lpa', salmax)

out.append(
    'last_active range: min=' +
    min(last_active) +
    ' max=' +
    max(last_active)
)

out.append(
    'crude_honeypot_heuristic_hits=' +
    str(honeypot_hits) +
    ' examples=' +
    str(hp_examples)
)

open(
    'data_profile.txt',
    'w',
    encoding='utf-8'
).write(chr(10).join(out))

print('wrote data_profile.txt; N=', n)

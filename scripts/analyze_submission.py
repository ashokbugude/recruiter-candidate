import csv,statistics
from collections import Counter
p='..\\team_sarva_automata.csv'
with open(p,encoding='utf-8') as f:
    r=list(csv.reader(f))
header=r[0]
rows=r[1:]
print('Header:',header)
print('Rows:',len(rows))
ids=[row[0] for row in rows]
ranks=[int(row[1]) for row in rows]
scores=[float(row[2]) for row in rows]
# checks
print('Unique ids:',len(set(ids))==len(ids))
print('Ranks 1..100:', set(ranks)==set(range(1,101)))
mono=True
for i in range(len(scores)-1):
    if scores[i] < scores[i+1]: mono=False; break
print('Scores non-increasing:',mono)
# score distribution
print('Score min/max/mean/median:',min(scores),max(scores),statistics.mean(scores),statistics.median(scores))
# reasoning analysis
reasons=[row[3] for row in rows]
lengths=[len(r.split()) for r in reasons]
print('Reasoning word len: min/median/max',min(lengths),statistics.median(lengths),max(lengths))
# top-10 reason uniqueness
print('Top-10 unique reason strings:', len(set(reasons[:10]))==10)
# detect endings
endings=[r[-60:] for r in reasons]
most_common=Counter(endings).most_common(5)
print('Most common reasoning endings (sample):')
for e,c in most_common[:5]: print(c,e[:120])
# repetition of exact templates
tmpl_counts=Counter(reasons)
most_common_r=tmpl_counts.most_common(10)
print('Top repeated reasoning templates (count, sample prefix):')
for r,c in most_common_r[:10]: print(c, r[:120])
# top-10 short checklist of signals for each top-10
print('\nTop-10 candidates and reasoning length:')
for i in range(10):
    print(i+1, rows[i][0], scores[i], lengths[i])

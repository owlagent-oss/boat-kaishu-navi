"""ボート回収率ナビ 毎日の自動処理。
  python3 navi.py predict   締切が近いレースの直前情報とオッズを取り、期待値を計算して data/picks/ に保存
  python3 navi.py settle    前日までの結果を取り込み、狙い目の成績(data/record.json)を更新
  python3 navi.py build     index.html を作る
"""
import datetime as dt, gzip, io, itertools, json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, lightgbm as lgb, lhafile
import parse

JST = dt.timezone(dt.timedelta(hours=9))
UA = {'User-Agent': 'Mozilla/5.0 (boat-kaishu-navi)'}
JCD = {'01':'桐生','02':'戸田','03':'江戸川','04':'平和島','05':'多摩川','06':'浜名湖','07':'蒲郡','08':'常滑',
       '09':'津','10':'三国','11':'びわこ','12':'住之江','13':'尼崎','14':'鳴門','15':'丸亀','16':'児島',
       '17':'宮島','18':'徳山','19':'下関','20':'若松','21':'芦屋','22':'福岡','23':'唐津','24':'大村'}
FEATS = ['waku', 'jcd_c', 'cls_c', 'age', 'weight', 'nat_win', 'nat_2', 'loc_win', 'loc_2', 'motor_2', 'boat_2',
         'exhibit', 'win_rank', 'motor_rank', 'ex_rank', 'win_gap', 'wind_m', 'wave_cm', 'b1_win', 'race_max_win']
# 狙い目のルール（過去の検証で決めたもの。変えるときは README に理由を書く）
EV_MAIN, ODDS_MAX = 1.2, 10.0   # 狙い目: 期待値1.2以上かつ単勝10倍以下
EV_SUB = 1.0                    # 参考: 期待値1.0以上
EV_MKT = 1.2                    # 単勝 新ルール（検証中）: 3連単オッズから出した1着確率 × 単勝オッズ（控除25%で換算）が1.2以上
EV_T3, T3_N = 1.2, 4            # 3連単: 単勝の期待値1.2以上の艇を1着に固定し、2・3着は確率の高い順に4点（基本）
T3_WIDE = 8                     # 広げるときは8点。ボックスは1着候補＋2・3着に来やすい2艇の3艇ボックス（6点）
BOAT = ['waku', 'cls_c', 'age', 'weight', 'nat_win', 'nat_2', 'loc_win', 'loc_2', 'motor_2', 'boat_2', 'exhibit',
        'win_rank', 'motor_rank', 'ex_rank', 'win_gap']
RACE = ['jcd_c', 'wind_m', 'wave_cm', 'race_max_win', 'b1_win']
F2 = BOAT + RACE + ['p1_waku', 'p1_win', 'p1_ex', 'p1_prob', 'prob']
F3 = F2 + ['p2_waku', 'p2_win', 'p2_ex', 'p2_prob']
T3ORDER = [f'{f}-{s}-{t}' for i in range(20) for f in range(1, 7)
           for s, t in [[(s, t) for s in range(1, 7) if s != f for t in range(1, 7) if t not in (f, s)][i]]]
WINDOW_MIN = (2, 25)            # 締切の2〜25分前のレースを処理する
PICKS = 'data/picks'

def now():
    return dt.datetime.now(JST)

def get(url, binary=False, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                b = r.read()
                return b if binary else b.decode('utf-8', 'replace')
        except Exception:
            time.sleep(2 + i * 3)
    return None

def lzh_text(url):
    b = get(url, binary=True)
    if not b:
        return None
    f = lhafile.LhaFile(io.BytesIO(b))
    return f.read(f.namelist()[0]).decode('cp932', 'replace')

# ---------- 番組表 ----------
def program(day):
    path = f'{PICKS}/{day:%Y%m%d}_program.csv'
    if os.path.exists(path):
        return pd.read_csv(path, dtype={'jcd': str, 'toban': str, 'motor_no': str, 'boat_no': str, 'deadline': str})
    t = lzh_text(f'https://www1.mbrace.or.jp/od2/B/{day:%Y%m}/b{day:%y%m%d}.lzh')
    if not t:
        return None
    p = pd.DataFrame(parse.parse_b(t, f'{day:%Y-%m-%d}'))
    os.makedirs(PICKS, exist_ok=True)
    p.to_csv(path, index=False)
    return p

# ---------- 直前情報・オッズ ----------
def beforeinfo(jcd, rno, day):
    h = get(f'https://boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd}&hd={day:%Y%m%d}')
    if not h:
        return None
    ex = {}
    for body in re.findall(r'<tbody class="is-fs12[^"]*">(.*?)</tbody>', h, re.S):
        w = re.search(r'is-boatColor(\d)', body)
        e = re.search(r'kg</td>\s*<td rowspan="4">([\d.]+)</td>', body)
        if w:
            ex[int(w.group(1))] = float(e.group(1)) if e else np.nan
    txt = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', h))
    wind = re.search(r'風速 (\d+)m', txt)
    wave = re.search(r'波高 (\d+)cm', txt)
    return dict(exhibit=ex, wind_m=int(wind.group(1)) if wind else np.nan, wave_cm=int(wave.group(1)) if wave else np.nan)

def tan_odds(jcd, rno, day):
    h = get(f'https://boatrace.jp/owpc/pc/race/oddstf?rno={rno}&jcd={jcd}&hd={day:%Y%m%d}')
    if not h:
        return None
    v = re.findall(r'oddsPoint[^>]*>([^<]+)<', h)
    out = {}
    for i, x in enumerate(v[:6], 1):
        try:
            out[i] = float(x)
        except ValueError:
            out[i] = 0.0
    return out if len(out) == 6 else None

# ---------- 特徴量・予測 ----------
_models = {}
def model(name='model.txt'):
    if name not in _models:
        _models[name] = lgb.Booster(model_file=name)
    return _models[name]

def t3_odds(jcd, rno, day):
    h = get(f'https://boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd}&hd={day:%Y%m%d}')
    v = re.findall(r'oddsPoint[^>]*>([^<]+)<', h or '')
    if len(v) != 120:
        return {}
    out = {}
    for c, x in zip(T3ORDER, v):
        try: out[c] = float(x)
        except ValueError: out[c] = 0.0
    return out

def t3_picks(g, a):
    """1着をa号艇に固定したときの、2・3着の確率が高い並び T3_N 点"""
    info = g.set_index('waku')
    def given(x, tag, w):
        x[f'{tag}_waku'] = w; x[f'{tag}_win'] = info.nat_win[w]; x[f'{tag}_ex'] = info.exhibit[w]; x[f'{tag}_prob'] = info.prob[w]
        return x
    x2 = given(g[g.waku != a].copy(), 'p1', a)
    q2 = model('m2.txt').predict(x2[F2]); q2 = dict(zip(x2.waku, q2 / q2.sum()))
    combos = []
    for b in q2:
        x3 = given(given(g[~g.waku.isin([a, b])].copy(), 'p1', a), 'p2', b)
        q3 = model('m3.txt').predict(x3[F3]); q3 = q3 / q3.sum()
        for c, v in zip(x3.waku, q3):
            combos.append((f'{a}-{b}-{c}', float(info.prob[a] * q2[b] * v)))
    combos.sort(key=lambda x: -x[1])
    m = {}
    for c, p in combos:
        _, b, c3 = map(int, c.split('-')); m[b] = m.get(b, 0) + p; m[c3] = m.get(c3, 0) + p
    box = [a] + sorted(m, key=m.get, reverse=True)[:2]
    return combos[:T3_WIDE], box

def features(g):
    g = g.copy()
    g['jcd_c'] = g.jcd.astype(int)
    g['cls_c'] = g.cls.map({'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1})
    g['win_rank'] = g.nat_win.rank(ascending=False, method='min')
    g['motor_rank'] = g.motor_2.rank(ascending=False, method='min')
    g['ex_rank'] = g.exhibit.rank(method='min')
    top2 = g.nat_win.nlargest(2)
    other = top2.iloc[-1] if len(top2) > 1 else np.nan
    g['win_gap'] = np.where(g.win_rank == 1, g.nat_win - other, g.nat_win - g.nat_win.max())
    g['race_max_win'] = g.nat_win.max()
    g['b1_win'] = g.sort_values('waku').nat_win.iloc[0]
    return g

def predict_race(g, day):
    jcd, rno = g.jcd.iloc[0], int(g.rno.iloc[0])
    bi, od = beforeinfo(jcd, rno, day), tan_odds(jcd, rno, day)
    if not bi or not od:
        return None
    g = g.copy()
    g['exhibit'] = g.waku.map(bi['exhibit'])
    g['wind_m'], g['wave_cm'] = bi['wind_m'], bi['wave_cm']
    g = features(g)
    raw = model().predict(g[FEATS])
    g['prob'] = raw / raw.sum()
    g['odds'] = g.waku.map(od)
    g['ev'] = np.where(g.odds > 0, g.prob * g.odds, 0)
    boats = []
    thin = (g.odds > 0).sum() < 5  # 票がまだ少なくオッズが出ていない艇が多い
    for r in g.sort_values('waku').itertuples():
        tag = '' if thin else ('main' if (r.ev >= EV_MAIN and 0 < r.odds <= ODDS_MAX) else ('sub' if r.ev >= EV_SUB else ''))
        boats.append(dict(waku=int(r.waku), name=r.name, cls=r.cls, prob=round(float(r.prob), 4),
                          odds=float(r.odds), ev=round(float(r.ev), 3), exhibit=None if pd.isna(r.exhibit) else float(r.exhibit), tag=tag))
    t3 = []
    firsts = [b['waku'] for b in boats if b['ev'] >= EV_T3 and b['odds'] > 0 and not thin]
    o3 = t3_odds(jcd, rno, day)  # 狙い目がなくてもオッズの動きを記録するため毎回取る
    # 単勝 新ルール: 3連単の市場が見ている1着確率と単勝オッズのズレ
    v3 = {c: x for c, x in (o3 or {}).items() if x > 0}
    if len(v3) >= 100 and not thin:
        s3 = sum(1 / x for x in v3.values()); m1 = {}
        for c, x in v3.items():
            m1[int(c[0])] = m1.get(int(c[0]), 0) + (1 / x) / s3
        st = sum(1 / b['odds'] for b in boats if b['odds'] > 0)
        for b in boats:
            if b['odds'] > 0:
                b['mp'] = round(m1.get(b['waku'], 0), 4)
                b['mev'] = round(m1.get(b['waku'], 0) * 0.75 / ((1 / b['odds']) / st), 3)
                b['mkt'] = b['mev'] >= EV_MKT
    if firsts:
        for a in firsts:
            cs, box = t3_picks(g, a)
            bc = ['-'.join(map(str, x)) for x in itertools.permutations(box)]
            t3.append(dict(first=a, combos=[dict(c=c, p=round(p, 4), odds=o3.get(c, 0.0)) for c, p in cs],
                           box=box, box_odds={c: o3.get(c, 0.0) for c in bc}))
    return dict(jcd=jcd, place=JCD.get(jcd, jcd), rno=rno, t3=t3, deadline=g.deadline.iloc[0], race_type=g.race_type.iloc[0],
                wind_m=None if pd.isna(bi['wind_m']) else int(bi['wind_m']), wave_cm=None if pd.isna(bi['wave_cm']) else int(bi['wave_cm']),
                fetched=now().strftime('%H:%M'), boats=boats,
                _snap=dict(tan=[od.get(i, 0.0) for i in range(1, 7)], t3=[o3.get(c, 0.0) for c in T3ORDER] if o3 else []))

SNAP = 'data/odds_ts'
def save_snap(t, snap):
    """オッズの動きを見るため、取った時点の単勝・3連単オッズを1回ごとに別ファイルで保存（3連単は T3ORDER の順）"""
    if not snap:
        return
    d = f'{SNAP}/{t:%Y%m%d}'
    os.makedirs(d, exist_ok=True)
    with gzip.open(f'{d}/{t:%H%M}.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(dict(at=t.strftime('%H:%M:%S'), races=snap), f, separators=(',', ':'))

def load_picks(day):
    path = f'{PICKS}/{day:%Y%m%d}.json'
    return json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}

def save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)

def cmd_predict(force_all=False):
    t = now(); day = t.date()
    p = program(day)
    if p is None or p.empty:
        print('番組表なし'); return
    picks = load_picks(day)
    todo = []
    for (jcd, rno), g in p.groupby(['jcd', 'rno']):
        if not isinstance(g.deadline.iloc[0], str) or ':' not in g.deadline.iloc[0]:
            continue
        hh, mm = map(int, g.deadline.iloc[0].split(':'))
        left = (dt.datetime.combine(day, dt.time(hh, mm), JST) - t).total_seconds() / 60
        if force_all and left > WINDOW_MIN[0] or WINDOW_MIN[0] <= left <= WINDOW_MIN[1]:
            todo.append(g)
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda g: predict_race(g, day), todo))
    n = 0; snap = {}
    for r in res:
        if r:
            k = f"{r['jcd']}-{r['rno']:02d}"
            snap[k] = dict(deadline=r['deadline'], **r.pop('_snap'))
            picks[k] = r; n += 1
    save_json(f'{PICKS}/{day:%Y%m%d}.json', picks)
    save_snap(t, snap)
    print(f'{t:%H:%M} 処理 {n}/{len(todo)}レース')

# ---------- 答え合わせ ----------
def cmd_settle():
    rec_path = 'data/record.json'
    rec = json.load(open(rec_path, encoding='utf-8')) if os.path.exists(rec_path) else {'days': {}}
    today = now().date()
    for f in sorted(os.listdir(PICKS)):
        if not re.fullmatch(r'\d{8}\.json', f):
            continue
        day = dt.datetime.strptime(f[:8], '%Y%m%d').date()
        if day >= today or f[:8] in rec['days']:
            continue
        t = lzh_text(f'https://www1.mbrace.or.jp/od2/K/{day:%Y%m}/k{day:%y%m%d}.lzh')
        if not t:
            continue
        _, _, pays = parse.parse_k(t, f'{day:%Y-%m-%d}')
        win = {(x['jcd'], x['rno']): (int(x['combo']), x['payout']) for x in pays if x['bet'] == '単勝'}
        win3 = {(x['jcd'], x['rno']): (x['combo'], x['payout']) for x in pays if x['bet'] == '3連単'}
        picks = json.load(open(f'{PICKS}/{f}', encoding='utf-8'))
        stat = {k: dict(bets=0, hits=0, ret=0) for k in ('main', 'sub', 't3', 't3_8', 'box', 'mkt')}
        for key, r in picks.items():
            w = win.get((r['jcd'], r['rno']))
            r['winner'] = w[0] if w else None
            w3 = win3.get((r['jcd'], r['rno']))
            r['result3'] = w3[0] if w3 else None
            if w3:
                for t in r.get('t3', []):
                    bets = {'t3': [c['c'] for c in t['combos'][:T3_N]], 't3_8': [c['c'] for c in t['combos'][:T3_WIDE]],
                            'box': ['-'.join(map(str, x)) for x in itertools.permutations(t.get('box', []))]}
                    for k, cs in bets.items():
                        stat[k]['bets'] += len(cs)
                        if w3[0] in cs:
                            stat[k]['hits'] += 1; stat[k]['ret'] += w3[1]
            for b in r['boats']:
                if b.get('mkt') and w:
                    stat['mkt']['bets'] += 1
                    if w[0] == b['waku']:
                        stat['mkt']['hits'] += 1; stat['mkt']['ret'] += w[1]
                for k in ('main', 'sub'):
                    if b['tag'] == 'main' or (k == 'sub' and b['tag'] == 'sub'):
                        if not w:
                            continue  # 中止・不成立は数えない
                        stat[k]['bets'] += 1
                        if w[0] == b['waku']:
                            stat[k]['hits'] += 1; stat[k]['ret'] += w[1]
        save_json(f'{PICKS}/{f}', picks)
        rec['days'][f[:8]] = stat
        print(day, stat)
    save_json(rec_path, rec)

# ---------- ページ ----------
def cmd_build():
    day = now().date()
    picks = load_picks(day)
    rec = json.load(open('data/record.json', encoding='utf-8')) if os.path.exists('data/record.json') else {'days': {}}
    tot = {k: dict(bets=0, hits=0, ret=0) for k in ('main', 'sub', 't3', 't3_8', 'box', 'mkt')}
    for s in rec['days'].values():
        for k in tot:
            for f in ('bets', 'hits', 'ret'):
                tot[k][f] += s.get(k, {}).get(f, 0)
    data = dict(date=f'{day:%Y-%m-%d}', updated=now().strftime('%m/%d %H:%M'), races=sorted(picks.values(), key=lambda r: r['deadline']),
                record=dict(total=tot, days=rec['days']), rule=dict(ev_main=EV_MAIN, odds_max=ODDS_MAX, ev_sub=EV_SUB, ev_t3=EV_T3, ev_mkt=EV_MKT, t3_n=T3_N, t3_wide=T3_WIDE))
    save_json('data/site.json', data)
    print('data/site.json', len(picks), 'レース')

if __name__ == '__main__':
    c = sys.argv[1] if len(sys.argv) > 1 else 'predict'
    {'predict': lambda: cmd_predict('--all' in sys.argv), 'settle': cmd_settle, 'build': cmd_build}[c]()

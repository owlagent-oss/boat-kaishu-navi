"""場ごとの基本データ（荒れランキング・平穏ランキング）を作る。
  python3 stats.py [開始日 終了日]   例: python3 stats.py 2024-11-01 2026-10-31
競走成績（K ファイル）を cache/K/ に保存して集計し、data/stats.json に書き出す。
荒れ度の基準（2026-11-13 奥ちゃんOK）: ①1号艇の1着率 ②3連単配当の中央値。両方の順位の平均で並べる。
"""
import datetime as dt, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import parse
from navi import JCD, get

CACHE = 'cache/K'

def fetch(day):
    path = f'{CACHE}/k{day:%y%m%d}.lzh'
    if not os.path.exists(path):
        b = get(f'https://www1.mbrace.or.jp/od2/K/{day:%Y%m}/k{day:%y%m%d}.lzh', binary=True)
        if not b:
            return None
        with open(path, 'wb') as f:
            f.write(b)
    return path

def main(start, end):
    os.makedirs(CACHE, exist_ok=True)
    days = [start + dt.timedelta(d) for d in range((end - start).days + 1)]
    with ThreadPoolExecutor(8) as ex:
        paths = [p for p in ex.map(fetch, days) if p]
    R, E, P = [], [], []
    for p in paths:
        d = os.path.basename(p)[1:7]; date = f'20{d[:2]}-{d[2:4]}-{d[4:]}'
        try:
            r, e, pa = parse.parse_k(parse.read_lzh(p), date); R += r; E += e; P += pa
        except Exception as ex_:
            print('失敗', p, ex_)
    races = pd.DataFrame(R); ent = pd.DataFrame(E); pay = pd.DataFrame(P)
    key = ['date', 'jcd', 'rno']
    # 1着の艇と進入コース（返還・不成立は除く）
    win = ent[ent['rank'].astype(str).str.strip().isin(['01', '1'])][key + ['waku', 'course']]
    t3 = pay[pay.bet == '3連単'][key + ['payout']].drop_duplicates(key)
    df = races[key].merge(win, on=key).merge(t3, on=key)
    df['b1'] = (df.waku == 1).astype(int)
    df['c1'] = (df.course.astype(str) == '1').astype(int)
    df['man'] = (df.payout >= 10000).astype(int)

    def summary(g):
        return dict(races=int(len(g)), b1_win=round(g.b1.mean() * 100, 1), c1_win=round(g.c1.mean() * 100, 1),
                    t3_median=int(g.payout.median()), manshu=round(g.man.mean() * 100, 1))
    places = []
    for jcd, g in df.groupby('jcd'):
        s = summary(g); s.update(jcd=jcd, name=JCD[jcd])
        s['course_win'] = [round((g.course.astype(str) == str(c)).mean() * 100, 1) for c in range(1, 7)]
        places.append(s)
    # 荒れ度: ①1号艇1着率が低いほど ②3連単中央値が高いほど 荒れる。順位の平均で決める
    pl = pd.DataFrame(places)
    pl['score'] = (pl.b1_win.rank(ascending=True) + pl.t3_median.rank(ascending=False)) / 2
    pl = pl.sort_values(['score', 'b1_win'])
    out = dict(period=f'{start:%Y-%m-%d}〜{end:%Y-%m-%d}', built=dt.datetime.now().strftime('%Y-%m-%d'),
               national=summary(df), places=pl.drop(columns='score').to_dict('records'))
    with open('data/stats.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('data/stats.json', len(df), 'レース', len(pl), '場', '取得', len(paths), '/', len(days), '日')

if __name__ == '__main__':
    a = sys.argv[1:] or ['2024-11-01', '2026-10-31']
    main(*(dt.date.fromisoformat(x) for x in a))

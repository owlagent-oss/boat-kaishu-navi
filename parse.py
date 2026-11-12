"""ボート回収率ナビ: 公式の競走成績(K)・番組表(B)のLZHをCSVに変換する。
使い方: python3 parse.py <rawディレクトリ> <出力ディレクトリ>
"""
import csv, glob, os, re, sys, unicodedata
import lhafile

def z2h(s):
    return unicodedata.normalize('NFKC', s)

def read_lzh(path):
    f = lhafile.LhaFile(path)
    return f.read(f.namelist()[0]).decode('cp932', errors='replace')

def num(s):
    s = s.strip()
    try:
        return float(s) if s else ''
    except ValueError:
        return ''

# ---------- 番組表 ----------
B_RACE = re.compile(r'^\s*(\d+)R\s+(\S+)\s+H(\d+)m')
def parse_b(text, date):
    rows = []
    jcd = rno = None
    for line in text.splitlines():
        m = re.match(r'^(\d\d)BBGN', line)
        if m:
            jcd = m.group(1); continue
        h = z2h(line)
        m = B_RACE.match(h)
        if m:
            rno = int(m.group(1)); rtype = m.group(2); dist = m.group(3)
            dl = re.search(r'締切予定(\d+:\d+)', h); deadline = dl.group(1) if dl else ''
            continue
        m = re.match(r'^([1-6]) (\d{4})(.{4})(\d{2})(.{2})(\d{2})([AB][12])(.*)$', line)
        if m and jcd and rno:
            s = m.group(8)
            rows.append(dict(date=date, jcd=jcd, rno=rno, race_type=rtype, distance=dist, deadline=deadline,
                waku=int(m.group(1)), toban=m.group(2), name=m.group(3).replace('　', ''),
                age=int(m.group(4)), branch=m.group(5).replace('　', ''), weight=int(m.group(6)),
                cls=m.group(7), nat_win=num(s[0:5]), nat_2=num(s[5:11]), loc_win=num(s[11:16]),
                loc_2=num(s[16:22]), motor_no=s[22:25].strip(), motor_2=num(s[25:31]),
                boat_no=s[31:34].strip(), boat_2=num(s[34:40])))
    return rows

# ---------- 競走成績 ----------
K_RACE = re.compile(r'^\s*(\d+)R\s+(.+?)\s+H(\d+)m\s+(\S+)\s+風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm')
K_ENTRY = re.compile(r'^\s+(\S+)\s+([1-6]) (\d{4}) (.{8})\s*(\d+)\s+(\d+)\s+([\d.]+)\s+(\d)\s+(\S+)\s+(\S+)?')
K_ENTRY2 = re.compile(r'^\s+([A-Z0-9]{2})\s+([1-6]) (\d{4}) ')
K_PAY = re.compile(r'^\s+(単勝|複勝|2連単|2連複|拡連複|3連単|3連複)?\s+([\d-]+)\s+(\d+)(?:\s+人気\s+(\d+))?')
def parse_k(text, date):
    races, entries, pays = [], [], []
    jcd = rno = None; cur_bet = None
    for line in text.splitlines():
        m = re.match(r'^(\d\d)KBGN', line)
        if m:
            jcd = m.group(1); rno = None; continue
        h = z2h(line)
        m = K_RACE.match(h)
        if m and jcd:
            rno = int(m.group(1)); cur_bet = None
            races.append(dict(date=date, jcd=jcd, rno=rno, race_name=m.group(2).strip(),
                distance=m.group(3), weather=m.group(4), wind_dir=m.group(5),
                wind_m=int(m.group(6)), wave_cm=int(m.group(7))))
            continue
        if not rno:
            continue
        m = K_ENTRY.match(h)
        if m:
            entries.append(dict(date=date, jcd=jcd, rno=rno, rank=m.group(1), waku=int(m.group(2)),
                toban=m.group(3), exhibit=num(m.group(7)), course=int(m.group(8)), st=m.group(9)))
            continue
        m = K_ENTRY2.match(h)  # 欠場など展示・進入が無い行
        if m:
            entries.append(dict(date=date, jcd=jcd, rno=rno, rank=m.group(1), waku=int(m.group(2)),
                toban=m.group(3), exhibit='', course='', st=''))
            continue
        m = K_PAY.match(h)
        if m and ('人気' in h or m.group(1) in ('単勝', '複勝') or cur_bet in ('単勝', '複勝')):
            if m.group(1):
                cur_bet = m.group(1)
            if cur_bet in ('単勝', '複勝'):
                # 複勝は1行に2つ並ぶ
                for c, p in re.findall(r'(\d)\s+(\d+)', h.split(cur_bet)[-1] if m.group(1) else h):
                    pays.append(dict(date=date, jcd=jcd, rno=rno, bet=cur_bet, combo=c, payout=int(p), pop=''))
            else:
                pays.append(dict(date=date, jcd=jcd, rno=rno, bet=cur_bet, combo=m.group(2),
                    payout=int(m.group(3)), pop=m.group(4) or ''))
    return races, entries, pays

def write(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def main(raw, out):
    os.makedirs(out, exist_ok=True)
    B, R, E, P = [], [], [], []
    bad = []
    for p in sorted(glob.glob(f'{raw}/B/*.lzh')):
        d = os.path.basename(p)[1:7]; date = f'20{d[:2]}-{d[2:4]}-{d[4:]}'
        try: B += parse_b(read_lzh(p), date)
        except Exception as e: bad.append((p, e))
    for p in sorted(glob.glob(f'{raw}/K/*.lzh')):
        d = os.path.basename(p)[1:7]; date = f'20{d[:2]}-{d[2:4]}-{d[4:]}'
        try:
            r, e, pa = parse_k(read_lzh(p), date); R += r; E += e; P += pa
        except Exception as e: bad.append((p, e))
    write(f'{out}/program.csv', B); write(f'{out}/races.csv', R)
    write(f'{out}/results.csv', E); write(f'{out}/payouts.csv', P)
    print(f'番組表 {len(B)}行 / レース {len(R)} / 着順 {len(E)}行 / 払戻 {len(P)}行 / 失敗 {len(bad)}')
    for b in bad[:10]: print(' ', b)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

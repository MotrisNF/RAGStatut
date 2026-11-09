import sys, re, xml.etree.ElementTree as ET
src, dst, kind = sys.argv[1:4]
root = ET.parse(src).getroot()
def text(e): return re.sub(r'\s+', ' ', ''.join(e.itertext())).strip()
out = []
if kind == 'consolidada':
    for b in root.iter('bloque'):
        vs = b.findall('version')
        if not vs: continue
        for e in vs[-1]:
            t = text(e)
            if t: out.append(t)
else:
    tx = max(root.iter('texto'), key=lambda e: len(''.join(e.itertext())))
    for e in tx:
        if e.tag == 'table':
            for r in e.iter('tr'):
                c = [text(x) for x in r]
                if any(c): out.append(' | '.join(c))
        else:
            t = text(e)
            if t: out.append(t)
open(dst, 'w', encoding='utf-8').write('\n\n'.join(out) + '\n')
print(dst.split('/')[-1], len(out), 'bloques', sum(map(len, out)), 'chars')

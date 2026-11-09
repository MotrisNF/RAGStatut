import sys,re
src=open(sys.argv[2]+'/ev.py',encoding='utf-8').read().split('import tqdm as _;')[0]
exec(src)
from src.models import Chunk,Document
H=re.compile(r'^Art[ií]culo \d+(?: bis)?\. +[A-ZÁÉÍÓÚ]',re.M)
HEAD={}
class Strict:
    def __init__(s,size,head): s.size,s.head=size,head; s.m=MdChunker(size)
    def chunk_documents(s,ds):
        out=[]
        for d in ds:
            st=sorted({0,*[m.start() for m in H.finditer(d.text)]})
            for a,b in zip(st,st[1:]+[len(d.text)]):
                seg=d.text[a:b]; title=seg.split('\n',1)[0] if H.match(seg) else ''
                for c in s.m.chunk_document(Document(file_path=d.file_path,text=seg)):
                    n=Chunk(file_path=d.file_path,first_character_index=c.first_character_index+a,last_character_index=c.last_character_index+a,text=c.text)
                    if s.head and title: HEAD[(n.file_path,n.first_character_index)]=title
                    out.append(n)
        return out
OT=BM25Index._chunk_tokens; OW=VectorIndex._window_chunk.__func__
BM25Index._chunk_tokens=staticmethod(lambda t,c:OT(t,c)+t.tokenize(HEAD.get((c.file_path,c.first_character_index),'')))
def W(cls,chs):
    r=OW(cls,chs)
    for w,c in zip(r,chs):
        h=HEAD.get((c.file_path,c.first_character_index))
        if h: w.windows=[h+'\n'+x for x in w.windows]
    return r
VectorIndex._window_chunk=classmethod(W)
def go(tag,ck):
    HEAD.clear(); ch,b,v=build(ck,True)
    L=[len(c.text) for c in ch]; print(f'## {tag}: {len(ch)} chunks, media {sum(L)//len(L)} chars',flush=True)
    for f in('none','doc+et','doc'): run('  filtro='+f,b,v,f)
    return b,v
go('actual (MdChunker 2000)',MdChunker(2000))
go('artículo mínimo anterior',ArtChunker(2000))
go('artículo estricto 2000',Strict(2000,False))
go('estricto 2000 + título',Strict(2000,True))
b,v=go('estricto 1000 + título',Strict(1000,True))
go('estricto 500 + título',Strict(500,True))
print('--- fallos top5 estricto 1000+título, doc+et')
for q,n,a in items:
    if not any(hit(c,n,a) for c in search(b,v,q,10,{n,E})[:5]): print('-',n,a,q)

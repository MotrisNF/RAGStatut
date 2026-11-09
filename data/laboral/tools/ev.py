import sys,re,json,unicodedata,glob,os,time
root=sys.argv[1]; tools=sys.argv[2]; os.chdir(root); sys.path[:0]=[root,tools]
from bank import Q,ALIAS,E
from src.reader import CorpusReader
from src.chunker import MdChunker
from src.bm25_index import BM25Index
from src.vector_index import VectorIndex
from src.tokenizer_bm import Tokenizer
docs=CorpusReader('data/raw/laboral').read().documents
name=lambda p:os.path.basename(p)[:-4]
D={name(d.file_path):d for d in docs}
ART=re.compile(r'^Art[ií]culo (\d+)(?: bis)?\.',re.M)
spans={}
for n,d in D.items():
    m=list(ART.finditer(d.text)); sp={}
    for i,x in enumerate(m):
        e=m[i+1].start() if i+1<len(m) else len(d.text)
        a=int(x.group(1))
        # el primer match de numero con mas texto gana (indices/anexos no)
        if a not in sp or e-x.start()>sp[a][1]-sp[a][0]: sp[a]=(x.start(),e)
    spans[n]=sp
items=[(q,n,a) for q,g in Q.items() for n,a in g.items()]
print(len(items),'pares pregunta-documento;',len(Q),'preguntas')
for q,n,a in items:
    assert all(x in spans[n] for x in a),(n,a)
def hit(c,n,arts):
    if name(c.file_path)!=n: return False
    for a in arts:
        s,e=spans[n][a]; o=min(e,c.last_character_index)-max(s,c.first_character_index)
        if o>0 and o>=0.3*min(c.last_character_index-c.first_character_index,e-s): return True
    return False
class ArtChunker(MdChunker):
    SEPARATORS=("\nArtículo ","\n\n","\n"," ")
    CUT_BEFORE=frozenset({"\nArtículo "})
STOP=frozenset("que cual cuales cuanto cuanta cuantos cuantas como donde cuando quien de la el los las un una unos unas y o a en por para con sin se me mi mis te tu su sus lo le les al del es son hay tengo tiene puedo pueden qué dice si ya mas algo".split())
ORIG=Tokenizer.tokenize; OS=Tokenizer.STOPWORDS
fold=lambda s:''.join(c for c in unicodedata.normalize('NFKD',s) if not unicodedata.combining(c))
def es_tok(self,t): return ORIG(self,fold(t))
def build(chunker,es):
    Tokenizer.tokenize=es_tok if es else ORIG
    Tokenizer.STOPWORDS=STOP|{'estatuto'} if es else OS
    ch=chunker.chunk_documents(docs)
    return ch,BM25Index.build(ch),VectorIndex.build(ch)
def search(b,v,q,k,allow,cand=30,alpha=.5):
    h=b.search(q,10**6)
    if allow: h=[x for x in h if name(x.chunk.file_path) in allow]
    h=h[:max(cand,k)]
    if not h: return []
    s=v.chunk_score(q); top=h[0].score
    return [c for _,c in sorted([((1-alpha)*x.score/top+alpha*s[x.chunk_id],x.chunk) for x in h],key=lambda t:-t[0])][:k]
def run(tag,b,v,filt):
    r={k:0 for k in(1,3,5,10)}; t=time.time()
    for q,n,a in items:
        allow={'none':None,'doc':{n},'doc+et':{n,E}}[filt]
        got=search(b,v,q,10,allow)
        for k in r: r[k]+=any(hit(c,n,a) for c in got[:k])
    print(f"{tag:46s}"+' '.join(f"R@{k}={r[k]/len(items):.2f}" for k in r),f"{(time.time()-t)/len(items)*1000:.0f}ms/q",flush=True)
import tqdm as _;
ch,b,v=build(MdChunker(2000),False); print('chunks base',len(ch))
for f in('none','doc+et','doc'): run('tal cual, filtro='+f,b,v,f)
ch2,b2,v2=build(MdChunker(2000),True)
for f in('none','doc+et','doc'): run('+tokenizador ES, filtro='+f,b2,v2,f)
ch3,b3,v3=build(ArtChunker(2000),True); print('chunks art',len(ch3),'empiezan en Artículo:',sum(c.text.startswith('Artículo') for c in ch3))
for f in('none','doc+et','doc'): run('+tok ES+chunk artículo, filtro='+f,b3,v3,f)
# fallos en la mejor variante
print('--- fallos top5 (doc+et, mejor variante)')
for q,n,a in items:
    got=search(b3,v3,q,10,{n,E})
    if not any(hit(c,n,a) for c in got[:5]): print('-',n,a,q)
# dataset oficial
out={'rag_questions':[]};un={'rag_questions':[]}
for i,(q,n,a) in enumerate(items):
    qq=f"{q} [{n}]"; qid=f"lab-{i}"
    src=[{'file_path':D[n].file_path,'first_character_index':spans[n][x][0],'last_character_index':spans[n][x][1]} for x in a]
    out['rag_questions'].append({'question_id':qid,'question':qq,'answer':'-','sources':src}); un['rag_questions'].append({'question_id':qid,'question':qq})
for d,o in(('AnsweredQuestions',out),('UnansweredQuestions',un)):
    os.makedirs(f'data/datasets/{d}',exist_ok=True); json.dump(o,open(f'data/datasets/{d}/laboral.json','w'),ensure_ascii=False)

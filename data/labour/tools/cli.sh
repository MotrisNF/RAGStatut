cd "$1"; PY=/sgoinfre/students/$USER/rag-venv/bin/python; R="data/raw/laboral"; I=data/processed
t(){ /usr/bin/time -f "   [%es, %MKB]" "$@"; }
echo "### index full"; t $PY -m src index --raw_dir $R --index_dir $I --max_chunk_size 2000 --full 2>&1 | grep -vE 'it/s|batch|file/s|chunk/s|\|.*\|' 
du -sh $I; ls -la $I | awk '{print $5,$9}'
echo "### index no-op"; t $PY -m src index --raw_dir $R --index_dir $I 2>&1 | tail -2
echo "### incremental (añado un doc)"; printf 'Artículo 1. Prueba incremental.\n\nTexto de prueba de un convenio nuevo sobre vacaciones.\n' > $R/convenio_tmp.txt; t $PY -m src index --raw_dir $R --index_dir $I 2>&1 | grep -E 'Incremental|chunks|\[' 
rm $R/convenio_tmp.txt; echo "### incremental (borro)"; t $PY -m src index --raw_dir $R --index_dir $I 2>&1 | grep -E 'Incremental|chunks|\['
Q="cuantos dias de vacaciones tengo"
for r in bm25 hybrid; do echo "### search $r"; t $PY -m src search "$Q" --k 5 --retriever $r --save_directory /dev/shm/nope_$$ 2>&1 | grep -E 'file_path|first_character|\[' ; done
echo "### search cached"; t $PY -m src search "$Q" --k 5 --save_directory /dev/shm/nope_$$ 2>&1 | grep -E '\[.*KB'
rm -rf /dev/shm/nope_$$
echo "### serve"; $PY -m src serve --port 18765 --index_dir $I --raw_dir $R >/dev/null 2>serve.log & P=$!
for i in $(seq 1 40); do curl -s -m1 localhost:18765/health >/dev/null && break; sleep 0.5; done
curl -s -m10 localhost:18765/health; echo
curl -s -m20 -G localhost:18765/search --data-urlencode "query=permiso por matrimonio" -d k=3 -w '\nHTTP %{http_code} %{time_total}s\n' | cut -c1-500
curl -s -m20 -G localhost:18765/search --data-urlencode "query=permiso por matrimonio" -d k=3 -d retriever=bm25 -o /dev/null -w 'bm25 HTTP %{http_code} %{time_total}s\n'
curl -s -m60 -X POST localhost:18765/index -w '\nPOST /index HTTP %{http_code} %{time_total}s\n' | cut -c1-300
kill $P; wait $P 2>/dev/null; grep -iE 'error|traceback' serve.log | head

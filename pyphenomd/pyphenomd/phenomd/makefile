HEADERS = phenomd.h ringdown_spectrum_fitting.h

default: phenomd.so

phenomd.o: phenomd.c $(HEADERS)
	gcc -std=c99 -fPIC -O1 -g     -c -o phenomd.o phenomd.c

phenomd.so: phenomd.o
	gcc -L/usr/lib -L/usr/local/lib  phenomd.o -lgsl  -std=c99 -shared -fPIC -O1 -g    -o phenomd.so

clean:
	-rm -f phenomd.o
	-rm -f phenomd.so

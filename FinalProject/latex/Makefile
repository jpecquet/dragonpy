TEX     := main
LATEX   := pdflatex
BIBER   := biber
FLAGS   := -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all clean distclean view

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex refs.bib
	$(LATEX) $(FLAGS) $(TEX)
	$(BIBER) $(TEX)
	$(LATEX) $(FLAGS) $(TEX)
	$(LATEX) $(FLAGS) $(TEX)

view: $(TEX).pdf
	open $(TEX).pdf

clean:
	rm -f $(TEX).aux $(TEX).bcf $(TEX).blg $(TEX).bbl $(TEX).log \
	      $(TEX).out $(TEX).run.xml $(TEX).toc $(TEX).lof $(TEX).lot \
	      $(TEX).fls $(TEX).fdb_latexmk $(TEX).synctex.gz

distclean: clean
	rm -f $(TEX).pdf

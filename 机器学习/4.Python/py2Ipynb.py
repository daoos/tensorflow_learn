import IPython.nbformat.current as nbf

nb = nbf.read(open('4.1.intro.py', 'r'), 'py')
nbf.write(nb, open('4.1.intro.ipynb', 'w'), 'ipynb')
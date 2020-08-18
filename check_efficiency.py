import stringdist
import re

def levenshtein(s, t):
    return stringdist.levenshtein(s, t)

def liczSkutecznoscAlgorytmow(wzorzec, rozpoznanyTekst):
    wyrazenie = re.compile(r'\s+')
    wzorzec = re.sub(wyrazenie, '', wzorzec)

    return skutecznosc = levenshtein(wzorzec, rozpoznanyTekst)

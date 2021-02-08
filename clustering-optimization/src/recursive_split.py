"""
charsplit made available for all.
"""

from charsplit import Splitter
import hunspell

spellchecker = hunspell.HunSpell('/usr/share/hunspell/{}.dic'.format('de_DE'),
                                 '/usr/share/hunspell/{}.aff'.format('de_DE'))
splitter = Splitter()


def recursive_split(glossary_term, nlp) -> str:
    ergebnis = []
    for word in glossary_term.split():
        # if word is stop word, no processing is necessary
        if word in nlp.Defaults.stop_words:
            ergebnis += [word]
        elif word[-1] == '-':
            ergebnis += [word]
        elif len(word) < 14:
            # check if it ends with s and remove it
            # if, after removing s, term is tagged as NOUN, then remove s
            # and return it as it is
            if word[-1] == 's' and len(word) > 1:
                if spellchecker.spell(word[:-1]):
                    ergebnis += [word[:-1]]
                else:
                    ergebnis += [word]
            else:
                ergebnis += [word]
        else:
            if splitter.split_compound(word)[0][0] == 0:
                ergebnis += [splitter.split_compound(word)[0][1]]
            # elif splitter.split_compound(word)[0][0] < 0.85:
            elif splitter.split_compound(word)[0][0] < -0.6:
                ergebnis += [word]
            else:
                ergebnis += [item[:-1] if item[-1] == 's' and nlp(item)[0].pos_ != 'PROPN' and spellchecker.spell(
                    item[:-1]) else item for
                             item in
                             splitter.split_compound(word)[0][1:]]
    output = [glossary_term, ' '.join(ergebnis)]
    if output[1] == glossary_term:
        return glossary_term
    O = []
    for x in output[1].split():
        O.append(recursive_split(x, nlp))
    return ' '.join(O)
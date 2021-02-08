import spacy
import pprint


def group_negative_list(lang, file_):
    if lang == 'DE':
        MODEL = "de_core_news_sm"
    else:
        MODEL = "en_core_web_sm"
    nlp = spacy.load(MODEL)

    # separate intro abbreviations, verbs, adjectives, prepositions ...
    # get the available pos tags
    f_ = open(file_, "r").readlines()
    f_ = [item[:-1] for item in f_]

    doc = nlp(' '.join(f_))
    # 1. Get all possible POS tags
    all_pos = []
    for token in doc:
        if token.pos_ not in all_pos:
            all_pos.append(token.pos_)

    # 2. Group items under those tags
    pos_dict = {}
    for pos in all_pos:
        pos_dict[pos] = [str(x) for x in doc if x.pos_ == pos]

    # pprint.pprint(pos_dict)
    return pos_dict


def main():
    group_negative_list('DE', "Black List DE.txt")


if __name__ == '__main__':
    main()

import re
import spacy
import pandas
import time
import pprint

pp = pprint.PrettyPrinter()

nlp = spacy.load("de_core_news_sm")


class TemplateValidator:
    def __init__(self):
        self._sentence = None
        self._tags_dict = None

    @property
    def sentences(self):
        return self._sentences

    @sentences.setter
    def sentences(self, sents):
        self._sentences = sents

    def tag_sentence(self, sentence):
        split_sentence = sentence.split()
        sentence_as_list = re.findall(r'[\w\-()\'\".°,]+(?=.)', sentence)
        doc = nlp(' '.join(split_sentence))
        response = {"SENTENCE": sentence, "CONFORMANT SEGMENT": '', "CONDITION": '', "OBJECT": '', 'DETAILS OBJECT': '',
                    "ACTOR": '', "OBJECT + COMPLEMENT": '', 'HEAD': split_sentence[0], "VERB": '', "MODAL VERB": '',
                    "SYSTEM_NAME": ''}
        conditional_word = ''
        conditional_words = ['Im Falle', 'Wenn', 'Sobald', 'Falls', 'Bei', 'Beim']
        modal_verbs = ['muss', 'müssen', 'soll', 'sollen', 'wird', 'werden']
        condition = False
        verb_details_in_beginning = False
        for word in conditional_words:
            if sentence.startswith(word):
                conditional_word = response['HEAD']
                condition = True
        # check for verb details at the beginning
        # they entail <System> and modal verb inversion
        # if exists, first word must be ADP, ADV or VERB
        if (doc[0].pos_ == 'ADP' or doc[0].pos_ == 'ADV' or doc[0].pos_ == 'VERB') and condition is False:
            verb_details_in_beginning = True

        # 2.
        second_or_third_path = False
        # detect which template is the subject of the sentence
        additional_string = ''
        if 'fähig sein' in sentence or 'die Möglichkeit bieten' in sentence:
            second_or_third_path = True
        if 'fähig sein' in sentence:
            additional_string = 'fähig sein'
        if 'die Möglichkeit bieten' in sentence:
            additional_string = 'die Möglichkeit bieten'
            response['ACTOR'] = ' '.join(
                re.findall(r'(?<=\w )\w+(?= die Möglichkeit)', sentence))
        lemmatized_sentence = []
        for token in doc:
            lemmatized_sentence += [token.lemma_]

        # two while loops, two indices;
        # break out of both of them when a valid modal verb is found
        # the first loop goes over doc, and skips punctuations etc
        # the second goes over sentence_as_list

        i = 0
        j = 0
        modal_verb_found = False
        details_verb_found = False
        while i < len(doc):
            while j < len(sentence_as_list):
                # skip punctuation encountered in doc
                if doc[i].is_punct:
                    i = i + 1
                # if skipped, doc and sentence_as_list should point at the same word

                if doc[i].text in modal_verbs:
                    if sentence_as_list[j][-1] == ',':
                        # continue replaced by i++ and j++
                        i = i + 1
                        j = j + 1
                        continue
                    response['MODAL VERB'] = sentence_as_list[j]
                    modal_verb_found = True
                    # if an ADP follows directly a muss, then the ADP + NC that follows is definitely a DETAILS VERB
                    if doc[i + 1].pos_ == 'ADP':
                        response['DETAILS VERB'] = str(doc[i + 1]) + ' ' + str(list(doc[i + 1:].noun_chunks)[0])
                        details_verb_found = True
                    # 3.
                    # the modal verb should not be preceded by a comma! otherwise we get false positives

                    if condition or verb_details_in_beginning:
                        try:
                            NP = list(doc[i:].noun_chunks)[0]
                        except IndexError:
                            # the sentence is not valid.
                            print('sentence invalid: ', doc.text)
                            return response
                        # response[' '.join(split_sentence[i+1:(i+len(NP)+1)])] = 'SYSTEM_NAME'
                        response['SYSTEM_NAME'] = str(NP)
                    else:
                        try:
                            NP = list(doc[:i + 1].noun_chunks)[0]
                            # response[' '.join(split_sentence[(i-len(NP)-1):i-1])] = 'SYSTEM_NAME'
                            response['SYSTEM_NAME'] = str(NP)
                        except IndexError:
                            return response
                    break  # there should be only one modal verb of interest
                # here i++ and j++

                j = j + 1
                i = i + 1
            if j == len(sentence_as_list):
                break
            if modal_verb_found:
                break

        # if no model verb found so far, it can't be found and therefore return false
        if not 'MODAL VERB' in response:
            return response
        if condition or verb_details_in_beginning:
            response['ANCHOR'] = ' '.join(re.findall(r'{0}.+?{1}'.format(
                response['MODAL VERB'], response['SYSTEM_NAME']
            ), sentence))
        else:
            if 'SYSTEM_NAME' in response:
                response['ANCHOR'] = ' '.join(re.findall(r'{0}.+?{1}'.format(
                    response['SYSTEM_NAME'], response['MODAL VERB']
                ), sentence))
            else:
                return response

        if response['ANCHOR'] in sentence:
            # find segment that precedes anchor and starts with 'Wenn'
            if conditional_word:

                if sentence.find(conditional_word) < sentence.find(response['ANCHOR']) and sentence.find(
                        conditional_word) == 0:
                    # to check if the sentence actually starts with Wenn, add:
                    # .. and sentence.find('Wenn') == 0:
                    anchor_preceded_by_conditional_block = True
                    # mark the segment from Wenn to ANCHOR as CONDITION
                    response['CONDITION'] = sentence[
                                            sentence.find(conditional_word):sentence.find(response['ANCHOR']) - 2]

            # get the first NC that precedes the anchor and mark it as Objekt

            # detect "multiple verbs" such as:
            # verwalten (erstellen, ändern, speichern, löschen, importieren, exportieren)

            # consider verb to be the word before any parentheses that come at the end
            # and append the parentheses content to it
            matches = re.findall(r'(\w+(\sund\s)?\w+)(\s\(.+\))?(\.)?$', sentence, re.MULTILINE)
            response['VERB'] = ''.join(matches[0][:-1])

            if second_or_third_path:
                match = ' '.join(re.findall(r'(?<={0}).+?(?={1})'.format(
                    additional_string, re.sub(r'\(', '\(', re.sub(r'\)', '\)', response['VERB']))
                ), sentence))

            else:
                match = ' '.join(re.findall(r'(?<={0}).+?(?={1})'.format(
                    replace_parentheses(response['ANCHOR']), replace_parentheses(response['VERB'])
                ), sentence))
            if match == '':
                # print('sentence with error:', sentence)
                return response
            to_be_trimmed = [' ', ', ', ' zu', ', zu', ',']
            for t in to_be_trimmed:
                if match.startswith(t):
                    match = match[len(t):]
                if match.endswith(t):
                    match = match[:-len(t)]
            response['OBJECT + COMPLEMENT'] = match
            if verb_details_in_beginning:
                response['DETAILS VERB'] = sentence[:sentence.find(response['ANCHOR'])]
                response['OBJECT + COMPLEMENT'] += response['DETAILS VERB']
        return response

    def tag_sentences(self):
        response = []
        for s in self.sentences:
            # response.append(self.tag_sentence(s))

            to_keep = ['SENTENCE', 'CONDITION', 'MODAL VERB', 'SYSTEM_NAME', 'ACTOR', 'OBJECT + COMPLEMENT', 'VERB']
            tmp = {key: self.tag_sentence(s)[key] for key in to_keep}
            response.append(tmp)
            pp.pprint(response)
        return response


def read_afos(file_):
    f = open(file_, 'r').read().splitlines()
    return f


def replace_parentheses(target):
    return re.sub(r'\(', '\(', re.sub(r'\)', '\)', re.sub(r'\[', '\[', re.sub(r'\]', '\]', target))))


tv = TemplateValidator()
# tv.sentences = read_afos("Afos_test.txt")
# tv.sentences = read_afos("test.txt")
# tv.sentences = read_afos("test_so_far.txt")
tv.sentences = read_afos("OPENCOSS_input.txt")

pandas.DataFrame(tv.tag_sentences(), columns=[
    'SENTENCE', 'CONDITION', 'MODAL VERB', 'SYSTEM_NAME', 'ACTOR', 'OBJECT + COMPLEMENT', 'VERB'
]).to_csv('output.csv', sep='\t')


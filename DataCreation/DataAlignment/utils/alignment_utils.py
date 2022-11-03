from bs4 import BeautifulSoup
import json
import requests
from nltk import tokenize

def get_question_subject(question):
  return ' as '.join(question.split('Who has ')[1].split(' as ')[:-1])


def read_parsed_wikipedia(input_file):
    total_data = list()
    with open(input_file, 'r') as f:
        for line in f.readlines():
            curr_data = json.loads(line.strip())
            total_data.append(fix_text(curr_data))
    return total_data


def fix_text(sample):
    text_splits = sample['text'].split("&gt;")
    test_splits_new = [text_splits[i].replace("&lt;/a", "") if i % 2 == 1 else text_splits[i].split("&lt;a")[0] for i in
                       range(len(text_splits))]
    text_new = " ".join(test_splits_new)
    sample['text'] = text_new
    return sample


def get_webpage_sentences(url, list_indices):

    if url[0] not in list_indices:
        return ""

    total_data = list()
    with open(list_indices[url[0]], 'r') as f:
        for line in f.readlines():
            curr_data = json.loads(line.strip())
            total_data.append(curr_data)

    for sample in total_data:
        sample_url = sample['url'].split("?")[0] + '/' + sample['title'].replace(" ", "_")
        if url[0] == sample_url:
            text_splits = sample['text'].split("&gt;")
            test_splits_new = [text_splits[i].replace("&lt;/a", "") if i % 2 == 1 else text_splits[i].split("&lt;a")[0]
                               for i in range(len(text_splits))]
            text_new = "".join(test_splits_new)
            return text_new

    return ""


def get_wikipedia_url(url, url_mappings):

    if (url.split("/")[-1] in url_mappings) and (url_mappings[url.split('/')[-1]] != '-1'):
        return [url_mappings[url.split('/')[-1]]]

    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'lxml')
    links = list()
    for link in soup.find_all('a'):
        if (type(link.get('href')) == str) and ("//en" in link.get('href')) and ("wikipedia" in link.get('href')):
            links.append(link.get('href'))
    return links


def plural_to_singular(word):

    if len(word) > 3 and word[-1].lower() == 's':
        word = word[:-1]
    return word


def get_aliases(entity):
    labels = dict()
    labels['label'] = entity.data['labels']['en']['value'] if 'en' in entity.data['labels'] else ""
    labels['aliases'] = list()
    if ('aliases' in entity.data) and ('en' in entity.data['aliases']):
        if 'en' in entity.data['aliases']:
            for sample in entity.data['aliases']['en']:
                labels['aliases'].append(sample['value'])

    return labels


def find_all_phrases(text: str, substring, check_numeric=False):
    """
    Checks if a substring appears in a string, and if so returns all the instances of this substring.
    If the substring is a number, checks it only if it is at least 3 digit long.
    """

    if type(substring) == str:
        found_instances = list()
    else:
        found_instances = dict()

    if '. ' in substring:
        old_substring = substring
        substring = substring.replace('. ', '._')
        text = text.replace(old_substring, substring)


    if len(text) == 0 or len(substring) == 0:
        return found_instances

    phrases = tokenize.sent_tokenize(text)
    for i in range(len(phrases)):

        # if is a number must be at least three digits long
        if type(substring) == str:
            if check_numeric and len(substring) < 3 and substring.isnumeric():
                return found_instances

        if type(substring) == str:
            if substring in phrases[i]:
                # returns the sentence and the one before to give it a bit of context
                if i > 0:
                    found_instances.append(phrases[i - 1].replace('._', '. ').capitalize() + ' ' + phrases[i].replace('._', '. ').capitalize())
                else:
                    found_instances.append(phrases[i].replace('._', '. ').capitalize())

        else:
            for string in substring:
                if string.lower() in phrases[i].lower() and string not in found_instances:
                    # returns the sentence and the one before to give it a bit of context
                    if i > 0:
                        found_instances[string.lower()] = phrases[i - 1].replace('._', '. ').capitalize() + ' ' + phrases[i].replace('._', '. ').capitalize()
                    else:
                        found_instances[string.lower()] = phrases[i].replace('._', '. ').capitalize()

    return found_instances

from DataCreation.DataAlignment.utils.properties_constants import COMP_PROPERTIES, MNLI_COMP, PROP_SENTENCES, MNLI_COMP_SENT


def rephrase_singular_mnli_basic(property):

    base_sentence = PROP_SENTENCES[property][1]

    if 'Who' in base_sentence and property != 'P2650':
        new_quest = base_sentence.replace('Who', '{}')
    elif 'What' in base_sentence and property != 'P2650':
        subject = base_sentence[5:].split(' had ')[0].split(' were ')[0]
        new_quest = base_sentence.replace(('What '+ subject).replace('  ', ' '), '{}')
    elif 'Which' in base_sentence:
        subject = base_sentence[5:].split(' had ')[0].split(' were ')[0].split(' ran ')[0]
        subject = subject.split(' {')[0] if '{}' in subject else subject
        new_quest = base_sentence.replace(('Which ' + subject).replace('  ', ' '), '{}')
    elif property == 'P2650':
        subject = 'organisations'
        new_quest = base_sentence.replace(('Who or what ' + subject).replace('  ', ' '), '{}')
    else:
        new_quest = ''

    return new_quest.replace('?', '.')


def rephrase_double_mnli_basic(property):

    base_sentence = PROP_SENTENCES[property][1]
    mnli_sentence = ' '.join(base_sentence.split(' ')[1:]).replace('?', '.')
    return mnli_sentence

def rephrase_mnli_basic(property, orig_entity, comp_entity):

    if PROP_SENTENCES[property][0] > 1:
        return rephrase_double_mnli_basic(property).format(orig_entity, comp_entity)
    else:
        return rephrase_singular_mnli_basic(property).format(orig_entity, comp_entity)


def rephrase_mnli_comp_only(property, comp_entity, orig_entity):

    found = False
    for key in MNLI_COMP_SENT:
        if property in key.split('##'):
            if MNLI_COMP_SENT[key][0] == 1:
                hypothesis = MNLI_COMP_SENT[key][1].format(comp_entity, orig_entity, MNLI_COMP[property])
            else:
                hypothesis = MNLI_COMP_SENT[key][1].format(comp_entity, MNLI_COMP[property], orig_entity)
            found = True
    if not found:
        hypothesis = "{} is {}'s {}".format(comp_entity, orig_entity, MNLI_COMP[property])

    return hypothesis



def rephrase_mnli_complex(property, comp_entity, orig_entity):

    if property in MNLI_COMP:
       return rephrase_mnli_comp_only(property, comp_entity, orig_entity)
    elif property in PROP_SENTENCES:
        return rephrase_mnli_basic(property, orig_entity, comp_entity)
    return ""


def rephrase_mnli(property: str, comp_entity: str, orig_entity: str, is_comp: bool = False):

    if is_comp:
        return rephrase_mnli_complex(property, comp_entity, orig_entity)
    else:
        return rephrase_mnli_basic(property, orig_entity, comp_entity)


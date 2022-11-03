from DataCreation.DataAlignment.utils.properties_constants import POSSIBLE_TYPES, PROP_SENTENCES, COMP_SENTENCES, COMP_PROPERTIES


def rephrase_singular_quest(question, property, comp_prop):

    base_question = ''
    for key in COMP_SENTENCES:
        if comp_prop in COMP_SENTENCES[key][0]:
            base_question = '{} are the {} {} '.format(COMP_SENTENCES[key][1][0], COMP_PROPERTIES[comp_prop], COMP_SENTENCES[key][1][1])
            break

    if 'Who' in question and property != 'P2650':
        new_quest = question.replace('Who', 'persons that')
    elif 'What' in question and property != 'P2650':
        subject = question[5:].split(' had ')[0].split(' were ')[0]
        new_quest = question.replace('What '+ subject.replace(' ', ''), subject.replace(' ', '') + ' that ')
    elif 'Which' in question:
        subject = question[5:].split(' had ')[0].split(' were ')[0]
        subject = subject.split(' {')[0] if '{}' in subject else subject
        new_quest = question.replace('Which ' + subject.replace(' ', ''), subject.replace(' ', '') + ' that ')
    elif property == 'P2650':
        subject = 'organisations'
        new_quest = question.replace('Who or what ' + subject.replace(' ', ''), subject.replace(' ', '') + ' that ')

    if len(base_question) == 0:
        return ''
    return (base_question + new_quest).replace('  ', ' ').replace(' is ', ' are ').replace(' was ', ' were ').replace(' has ', ' have ')



def rephrase_subtyped_quest(question, comp_prop, subtype):

    splitter = question.split(subtype)[1]
    base_question = ''
    for key in COMP_SENTENCES:
        if comp_prop in COMP_SENTENCES[key][0]:
            base_question = '{} are the {} {} '.format(COMP_SENTENCES[key][1][0], COMP_PROPERTIES[comp_prop], COMP_SENTENCES[key][1][1])
            break
    if len(base_question) == 0:
        return ''
    question = base_question + subtype + ' that ' + splitter
    return question.replace(' is ', ' are ').replace(' was ', ' were ').replace(' has ', ' have ')


def rephrase_quest_composition(question, property, comp_prop, subtype):

    if PROP_SENTENCES[property][0] == 2:
        new_quest = rephrase_subtyped_quest(question, comp_prop, subtype)
    else:
        new_quest = rephrase_singular_quest(question, property, comp_prop)

    return new_quest


def rephrase_quest_intersec(question: str):

    type_in = ''
    for tp in POSSIBLE_TYPES:
        if tp in question and len(tp) > len(type_in):
            type_in = tp
    if 'in ' or 'to ' in question.lower():
        return ' '.join(question.split(' ')[2:]).replace(type_in, '')
    else:
        return ' '.join(question.split(' ')[1:]).replace(type_in, '')



def reformulate_questions_intersec(q1: str, q2:str, subject: str):

    """
    Receives two questions that have common answers and reformulates them as one.
    """
    if 'who' in q2.lower():
        new_quest = q1.replace('?', '') + ' and ' + q2[4:]
    else:
        new_quest = q1.replace('?', '') + ' and ' + q2.split(subject)[-1]
    return new_quest.replace('  ', ' ')


def filter_answer(answer, harsh=True, thresh=0.8):

  if answer[1] == 0:
    return False
  if answer[0] < 5:
    return False
  if answer[2] > 300:
    return False
  if (float(answer[0]) / answer[1] < thresh) and not harsh:
    return False
  if (float(answer[0]) / answer[2] < thresh) and harsh:
    return False
  return True
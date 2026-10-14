import re

def rule(latest,b):
  if b in ['gratitude','pride'] and re.search(r'松了一口气|释然|轻松(?:多了|一些|了些)|没那么堵|心里.*石头.*轻|不用再.*猜',latest):return 'relaxed'
  if b=='fear' and re.search(r'万一|会不会|最担心',latest) and not re.search(r'医生|体检|确诊|癌|刀|打我|死亡|危险',latest):return 'anxiety'
  if b=='anxiety' and re.search(r'无能为力|无路可走|怎么选都不对|根本没办法|不管.*都没用',latest):return 'helplessness'
  return b

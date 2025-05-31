import pandas as pd

# Load tagged comments
df = pd.read_csv('data_sets/tagged_comments.csv')

# Define weak rules
against_keywords = [
    'trădare', 'frustrare', 'plagiat', 'rușine', 'hoț', 'nesimțit', 'minciună',
    'dezamăgire', 'furat', 'corupt', 'dezamăgit', 'groaznic', 'catastrofă',
    'fals', 'incompetent', 'dictator', 'manipulare', 'mincinos', 'eşec',
    'dezgustător', 'abuz', 'nedreptate', 'penibil', 'indignare', 'mizerabil',
    'jenant', 'faliment', 'prostie', 'bătaie de joc', 'mafiot', 'toxic',
    'murdar', 'parazit', 'periculos', 'rău', 'scandal', 'păcat', 'vina',
    'umilit', 'josnic', 'ură', 'urăsc', 'dezordine', 'haos', 'necinstit',
    'înșelător', 'lovitură', 'pierdere', 'greșeală', 'vinovat', 'slab',
    'șantaj', 'arogant', 'distrus', 'răzbunare', 'pedepsit', 'odioasă',
    'infam', 'batjocură', 'criminal', 'furt', 'nelegiuit', 'degradat',
    'abject', 'ipocrit', 'nenorocire', 'pierzător', 'blam', 'blestem',
    'catastrofal', 'neiertat', 'penal', 'absurd', 'perfid', 'rebut',
    'ineficient', 'întârziat', 'blocaj', 'oprit', 'neacceptat', 'amendat',
    'înjurat', 'deturnat', 'sabotor', 'ratat', 'praf', 'leneș', 'infestare',
    'coruptibil', 'șocant', 'nedorit', 'parșiv', 'prăbușire', 'înfrânt',
    'defect', 'viciat', 'exclus', 'expulzat', 'șubred', 'păcălit', 'condamnat',
    'împiedicat', 'mizerie', 'dezgust', 'moarte', 'morbid', 'lăcomie',
    'nepriceput', 'obosit', 'umflat', 'ridicol', 'neîncredere', 'îndoială',
    'dificil', 'lipsă', 'nefuncțional', 'nerealizat', 'înfrângere', 'dezinteres',
    'cenzură', 'persecutat', 'demis', 'anulat', 'respins', 'refuzat', 'părăsit',
    'pierderi', 'înșelat', 'pedeapsă', 'mort', 'învins', 'rănit', 'inacceptabil',
    'nefolositor', 'pierdut', 'nesigur', 'întuneric', 'pătat', 'disprețuit',
    'aruncat', 'ruinat', 'demolat', 'slăbit', 'ruinat', 'stricat', 'viciu',
    'șubrezit', 'dezafectat', 'abandonat', 'crimă', 'pericol', 'falsificat',
    'dezvăluire', 'acuzat', 'defăimat', 'ostil', 'acuzație', 'denunț', 'reprimat',
    'dictatură', 'derapaj', 'alarmant', 'dezertare', 'cădere', 'prăpastie',
    'frică', 'temere', 'amenințare', 'conspirație', 'revoltă', 'păgubos',
    'inapt', 'nepricepere', 'scădere', 'impunitate', 'aroganță', 'dezbinare',
    'divizare', 'corupție', 'morală scăzută', 'prăpădit', 'lene', 'trucaj',
    'decepție', 'delapidare', 'infractor', 'mașinațiune', 'nedrept', 'abandon'
]


for_keywords = [
    'mulțumim', 'respect', 'felicitări', 'bine', 'susțin', 'bun', 'stabilitate',
    'sprijin', 'bravo', 'excelent', 'superb', 'corect', 'demnitate', 'apreciat',
    'remarcabil', 'onest', 'cinstit', 'transparent', 'valoros', 'iubit',
    'deosebit', 'aplaud', 'mândru', 'puternic', 'echilibru', 'claritate',
    'ordine', 'progres', 'modernizare', 'succes', 'realizat', 'strălucitor',
    'eficient', 'minunat', 'frumos', 'plăcut', 'admirabil', 'dedicat', 'angajat',
    'curajos', 'motivat', 'sclipitor', 'inspirat', 'inteligent', 'educat',
    'rafinat', 'reformat', 'restructurat', 'reînnoit', 'revigorat', 'eroic',
    'salvator', 'vizionar', 'lider', 'strategic', 'brilliant', 'eficace',
    'magnific', 'inovator', 'câștigător', 'medaliat', 'campion', 'neînfricat',
    'glorios', 'victorie', 'curat', 'solid', 'integru', 'fericit', 'calm',
    'echilibrat', 'liniștit', 'pace', 'unitate', 'prietenos', 'altruist',
    'generos', 'optimist', 'energic', 'luminos', 'înțelept', 'carismatic',
    'împăciuitor', 'drept', 'just', 'legal', 'merituos', 'impecabil', 'solidar',
    'încurajator', 'respectat', 'activ', 'alert', 'ambițios', 'asumat',
    'autentic', 'binecuvântat', 'binevoitor', 'clarvăzător', 'competent',
    'consecvent', 'conștient', 'constructiv', 'convins', 'cultivat', 'deschis',
    'dornic', 'echitabil', 'entuziasm', 'evoluție', 'fiabil', 'formidabil',
    'glorificat', 'iluminat', 'implicat', 'înalt', 'inclusiv', 'inspirator',
    'merit', 'nobil', 'pasionat', 'plin de viață', 'pozitiv', 'prudent',
    'răbdător', 'răsplătit', 'recunoscut', 'refăcut', 'renovat', 'renăscut',
    'rezistent', 'revoluționar', 'serios', 'sincer', 'speranță', 'stăpânit',
    'strălucire', 'tenace', 'unificat', 'valorificat', 'valoros', 'venerabil',
    'verificat', 'victorios', 'vigilență', 'vital', 'voinic', 'voios', 'zelos',
    'salutat', 'mulțumit', 'mulțumitor', 'onorabil', 'apreciere', 'susținere',
    'alegere bună', 'evoluat', 'avansat', 'pozitivist', 'luminos', 'strălucit',
    'erou', 'echitate', 'emblematic', 'eminent', 'faimos', 'popular', 'renumit',
    'adorabil', 'adorat', 'acceptat', 'admirat', 'agreat', 'aprobat', 'ascultat',
    'binefăcător', 'bineînțeles', 'binevoință', 'candidat bun', 'clarificare',
    'curaj', 'dedicare', 'delicat', 'diplomat', 'distins', 'emblemă', 'emulație',
    'entuziasm', 'excelență', 'fidelitate', 'garanție', 'glorie', 'ideal',
    'inimă bună', 'model', 'neclintit', 'onorabil', 'ordine', 'perseverență',
    'promisiune', 'prosperitate', 'respectabil', 'reputație', 'rezolvat',
    'simbol', 'sinceritate', 'sprijinitor', 'star', 'talent', 'valoare',
    'virtute', 'voință', 'zâmbet', 'zest'
]


def weak_label(text):
    text = str(text).lower()
    if any(word in text for word in against_keywords):
        return 0  # against
    if any(word in text for word in for_keywords):
        return 1  # for
    return 2  # neutral

# Apply weak labeling
df['weak_label'] = df['text'].apply(weak_label)

# Filter only comments with Iohannis keywords (optional)
df_filtered = df[df['has_iohannis_keyword']]

# Save weakly labeled data
df_filtered[['text', 'weak_label']].to_csv('data_sets/weakly_labeled_comments_new_2.csv', index=False)
print(f"✅ Weakly labeled comments saved to 'weakly_labeled_comments.csv'")

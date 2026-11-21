"""Build the Sahayak Bangla seed dataset (prompt/completion, Bengali) + a held-out test set.

python src/build_seed.py  ->  data/seed_train.jsonl, data/test_heldout.jsonl, data/stats.txt
Every answer is composed only from facts/cards_bn.json (official myScheme text, checked 2026-11-16) and from the
fixed problem-solving notes below, so nothing is invented.
"""
import json, os, random, re
from rules import check

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
CARDS = {k: v for k, v in json.load(open(os.path.join(ROOT, 'facts', 'cards_bn.json'), encoding='utf-8')).items()
         if not k.startswith('_')}
BN = str.maketrans('0123456789', '০১২৩৪৫৬৭৮৯')
bn = lambda x: str(x).translate(BN)


def inr(n):
    """Indian digit grouping: 140000 -> ১,৪০,০০০"""
    s = str(int(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = ','.join([head[max(0, i - 2):i] for i in range(len(head), 0, -2)][::-1])
        s = head + ',' + tail
    return bn(s)

SYSTEM = ('তুমি "সহায়ক বাংলা" — পশ্চিমবঙ্গের মানুষদের সরকারি প্রকল্প বোঝাতে সাহায্য করো। সহজ চলিত বাংলায়, ছোট বাক্যে '
          'উত্তর দাও। টাকার পরিমাণ বা নিয়ম বানিয়ে বলবে না; নিশ্চিত না হলে অফিস বা সরকারি পোর্টালে যাচাই করতে বলবে। '
          'কখনো OTP বা টাকা কাউকে দিতে বলবে না।')

CLOSE = ['নিয়ম মাঝে মাঝে বদলায়, তাই আবেদনের আগে অফিস বা সরকারি পোর্টালে একবার মিলিয়ে নিন।',
         'সর্বশেষ নিয়ম জানতে BDO অফিস বা দুয়ারে সরকার ক্যাম্পে জিজ্ঞেস করে নেবেন।',
         'আবেদন সম্পূর্ণ বিনামূল্যে; কেউ টাকা চাইলে দেবেন না। সর্বশেষ তথ্য অফিসে মিলিয়ে নিন।',
         'কোনো দালালকে টাকা বা OTP দেবেন না। প্রয়োজনে সরাসরি সরকারি অফিসে যান।']

MEN = ['রহিম', 'সুবল', 'গোপাল', 'আনোয়ার', 'বিকাশ', 'মদন', 'সঞ্জয়', 'হারাধন', 'জাকির', 'প্রদীপ', 'শ্যামল', 'ইমরান']
WOMEN = ['মালতী', 'রুমা', 'সাবিনা', 'পূর্ণিমা', 'আশা', 'ফতেমা', 'লক্ষ্মী', 'মিনতি', 'রেহানা', 'সুমিত্রা', 'কাকলি', 'নাসরিন']
DISTRICTS = ['মুর্শিদাবাদ', 'পুরুলিয়া', 'বাঁকুড়া', 'নদিয়া', 'মালদা', 'উত্তর দিনাজপুর', 'দক্ষিণ ২৪ পরগনা', 'পূর্ব মেদিনীপুর',
             'বীরভূম', 'জলপাইগুড়ি', 'হুগলি', 'কোচবিহার', 'পশ্চিম বর্ধমান', 'ঝাড়গ্রাম']
CASTE_BN = {'sc': 'তফসিলি জাতি (SC)', 'st': 'তফসিলি উপজাতি (ST)', 'obc': 'OBC', 'gen': 'সাধারণ'}


VOWEL_SIGNS = set('ািীুূৃেৈোৌঅআইঈউঊঋএঐওঔ')


def gen(n):
    """Bengali possessive: প্রকল্প -> প্রকল্পের, যোজনা -> যোজনার, names ending in ')' -> '-এর'."""
    last = n.strip()[-1]
    if last in VOWEL_SIGNS:
        return n + 'র'
    if 'ক' <= last <= 'হ' or last in 'ড়ঢ়য়ৎ':
        return n + 'ের' if last != 'ৎ' else n + 'এর'
    return n + '-এর'


def loc(n):
    """Bengali locative: প্রকল্প -> প্রকল্পে, যোজনা -> যোজনায়, সাথী -> সাথীতে; names ending in ')' keep '-এ'."""
    last = n.strip()[-1]
    if last == 'া':
        return n + 'য়'
    if last in VOWEL_SIGNS:
        return n + 'তে'
    if 'ক' <= last <= 'হ' or last in 'ড়ঢ়য়':
        return n + 'ে'
    return n + '-এ'


def fix_gen(text, names):
    for nm in names:
        text = text.replace(nm + '-এর', gen(nm)).replace(nm + '-এ', loc(nm))
    return text


def bullet(items):
    return '\n'.join('• ' + x for x in items)


# ---------------------------------------------------------------- answer builders
def ans_overview(c, rnd):
    parts = [f"{c['name']} সম্পর্কে সংক্ষেপে:", f"কী পাওয়া যায়: {c['what']}", 'কারা পাবেন:\n' + bullet(c['who']),
             f"কোথায় আবেদন: {c['where']}"]
    if c.get('note'):
        parts.append(c['note'])
    parts.append(rnd.choice(CLOSE))
    return '\n\n'.join(parts)


def ans_amount(c, rnd):
    s = f"{c['name']}-এ: {c['what']}"
    if c.get('note'):
        s += '\n\n' + c['note']
    return s + '\n\n' + rnd.choice(CLOSE)


def ans_docs(c, rnd):
    return (f"{c['name']}-এর জন্য সাধারণত এই কাগজগুলো লাগে:\n{bullet(c['docs'])}\n\n"
            f"সব কাগজের নাম যেন আধারের নামের সঙ্গে হুবহু মেলে — না মিললে আগে সংশোধন করিয়ে নিন।\n\n{rnd.choice(CLOSE)}")


def ans_where(c, rnd):
    return f"{c['name']}-এর আবেদন: {c['where']}\n\nজমা দেওয়ার পরে রসিদ বা Application ID যত্ন করে রেখে দিন।\n\n{rnd.choice(CLOSE)}"


def ans_who(c, rnd):
    return f"{c['name']} পাওয়ার প্রধান শর্তগুলো:\n{bullet(c['who'])}\n\n{rnd.choice(CLOSE)}"


def ans_persona(slug, c, p, rnd):
    verdict, ok, no = check(slug, p)
    if verdict == 'no':
        head = f"আপনার কথা শুনে মনে হচ্ছে {c['name']} এখন পাওয়া কঠিন, কারণ:\n{bullet(no)}"
        if ok:
            head += f"\n\nযে শর্তগুলো মিলছে:\n{bullet(ok)}"
        tail = 'আপনার অবস্থা অনুযায়ী অন্য কোনো প্রকল্প মেলে কিনা BDO অফিস বা দুয়ারে সরকার ক্যাম্পে জিজ্ঞেস করুন।'
    else:
        head = (f"আপনার কথা অনুযায়ী {c['name']}-এর প্রধান শর্তগুলো মিলছে:\n{bullet(ok)}\n\n"
                f"কী পাবেন: {c['what']}\n\nকাগজপত্র:\n{bullet(c['docs'][:5])}\n\nকোথায়: {c['where']}")
        tail = 'বাকি শর্তগুলো (আয়, বাসস্থান ইত্যাদি) অফিস যাচাই করবে।'
    return f"{head}\n\n{tail}\n\n{rnd.choice(CLOSE)}", verdict


# ---------------------------------------------------------------- problem solving (general, safe advice)
PROBLEMS = [
    (['{s}-এর টাকা আসছে না, কী করব?', '{s} পাস হয়েছে বলছে কিন্তু ব্যাংকে টাকা ঢোকেনি।', '{s}-এর কিস্তি অনেক মাস বন্ধ।'],
     'টাকা না আসার সাধারণ কারণ ও সমাধান:\n'
     '• ব্যাংক অ্যাকাউন্টের সঙ্গে আধার লিংক (seeding) আছে কিনা ব্যাংক শাখায় গিয়ে দেখুন; না থাকলে সেখানেই আধার লিংক ও NPCI ম্যাপিং করিয়ে নিন।\n'
     '• অ্যাকাউন্ট বন্ধ বা নিষ্ক্রিয় হয়ে থাকলে KYC আপডেট করুন।\n'
     '• আবেদনের Application ID দিয়ে পোর্টালে বা অফিসে স্ট্যাটাস জিজ্ঞেস করুন।\n'
     '• তাতেও না হলে BDO অফিসে লিখিত অভিযোগ দিন এবং রসিদ রাখুন।'),
    (['আধারে আমার নাম আর ব্যাংকের খাতায় নাম আলাদা, {s} আটকে গেছে।', 'নামের বানান ভুল, {s} রিজেক্ট হয়ে গেল।'],
     'নাম না মিললে টাকা আটকে যায়। যা করবেন:\n'
     '• কোনটা ঠিক নাম ঠিক করে নিন (সাধারণত সার্টিফিকেট বা ভোটার কার্ডে যেটা আছে)।\n'
     '• আধারে ভুল থাকলে নিকটবর্তী আধার সেবা কেন্দ্রে সংশোধন করুন।\n'
     '• ব্যাংকে ভুল থাকলে শাখায় গিয়ে KYC আপডেট করে নাম ঠিক করান।\n'
     '• সংশোধনের পরে নতুন কপি নিয়ে আবার অফিসে জমা দিন।'),
    (['আধারে OTP আসছে না, {s}-এর ফর্ম ভরতে পারছি না।', '{s} অনলাইন আবেদনে OTP আসে না।'],
     'OTP না এলে:\n'
     '• আধারের সঙ্গে কোন মোবাইল নম্বর যুক্ত আছে দেখুন — পুরনো বা বন্ধ নম্বর হলে আধার সেবা কেন্দ্রে নতুন নম্বর যুক্ত করুন।\n'
     '• নেটওয়ার্ক ভালো জায়গায় কয়েক মিনিট পরে আবার চেষ্টা করুন।\n'
     '• অনলাইন সম্ভব না হলে সরাসরি অফিসে বা ক্যাম্পে কাগজে আবেদন করুন।\n'
     'মনে রাখবেন: OTP শুধু নিজে ফর্মে লিখবেন, অন্য কাউকে ফোনে বলবেন না।'),
    (['{s}-এর আবেদন করেছি, স্ট্যাটাস কীভাবে দেখব?', '{s} কতদূর হলো জানব কীভাবে?'],
     'স্ট্যাটাস দেখতে:\n'
     '• আবেদনের রসিদ বা Application ID হাতে রাখুন।\n'
     '• প্রকল্পের সরকারি পোর্টালে স্ট্যাটাস দেখার অংশে ID দিয়ে দেখুন।\n'
     '• অনলাইনে না পারলে যে অফিসে জমা দিয়েছিলেন (BDO / SDO / পঞ্চায়েত / স্কুল) সেখানে ID দেখিয়ে জিজ্ঞেস করুন।'),
    (['{s}-এর আবেদন বাতিল হয়ে গেছে, এখন কী করব?', '{s} রিজেক্ট হয়েছে, আবার কি করা যাবে?'],
     'আবেদন বাতিল হলে:\n'
     '• কেন বাতিল হয়েছে সেটা অফিস থেকে জেনে নিন (সম্ভব হলে লিখিত)।\n'
     '• কাগজ কম বা ভুল থাকলে ঠিক করে আবার আবেদন করুন।\n'
     '• শর্ত না মিললে অন্য কোনো প্রকল্প মেলে কিনা জিজ্ঞেস করুন।\n'
     '• মনে হলে ভুল করে বাতিল হয়েছে, BDO / SDO অফিসে লিখিত আবেদন করুন।'),
    (['একজন লোক বলছে টাকা দিলে {s} তাড়াতাড়ি করিয়ে দেবে, দেব?', 'দালাল {s}-এর জন্য ৫০০ টাকা চাইছে।', 'ফোনে বলছে {s}-এর জন্য OTP বলুন।'],
     'না, দেবেন না।\n'
     '• সরকারি প্রকল্পের আবেদন বিনামূল্যে — কাউকে টাকা দিতে হয় না।\n'
     '• OTP, ATM পিন বা ব্যাংকের তথ্য কখনো কাউকে বলবেন না, সরকারি কর্মীরাও ফোনে এগুলো চান না।\n'
     '• প্রতারণার চেষ্টা হলে ১৯৩০ নম্বরে ফোন করুন বা cybercrime.gov.in-এ অভিযোগ করুন।\n'
     '• আবেদন নিজে অফিসে বা দুয়ারে সরকার ক্যাম্পে জমা দিন।'),
]

# ---------------------------------------------------------------- question templates
Q = {
    'overview': ['{n} কী?', '{n} সম্পর্কে একটু বুঝিয়ে বলুন।', '{n} প্রকল্পটা কী জিনিস?', 'আমাকে {n}-এর ব্যাপারে বলবেন?'],
    'amount': ['{n}-এ কত টাকা পাওয়া যায়?', '{n}-এ কী সুবিধা পাওয়া যায়?', '{n} পেলে মাসে বা বছরে কত টাকা আসে?'],
    'docs': ['{n}-এর জন্য কী কী কাগজ লাগবে?', '{n}-এ আবেদন করতে কোন ডকুমেন্ট লাগে?', '{n}-এর ফর্মের সঙ্গে কী জমা দিতে হবে?'],
    'where': ['{n}-এর আবেদন কোথায় করব?', '{n}-এর ফর্ম কোথায় পাওয়া যায়?', '{n} কীভাবে অ্যাপ্লাই করব?'],
    'who': ['{n} কারা পায়?', '{n}-এর যোগ্যতা কী?', 'কে কে {n} পেতে পারে?'],
}
Q_TEST = {   # wordings used ONLY in the held-out test set
    'overview': ['{n} নিয়ে কিছু জানেন? একটু বলবেন।'],
    'amount': ['{n} থেকে টাকা কত মেলে বলুন তো?'],
    'docs': ['{n} করতে গেলে সঙ্গে কী কী নিয়ে যাব?'],
    'where': ['{n}-এর জন্য কোন অফিসে যেতে হবে?'],
    'who': ['{n} পাওয়ার শর্তগুলো কী কী?'],
}
BANGLISH_Q_TEST = {'amount': ['{b} te mase ba bochore koto pai?'], 'docs': ['{b} korte kon kon document niye jabo?'],
                   'where': ['{b} er form kothay jama dibo?'], 'who': ['{b} pete ki ki condition?']}
BANGLISH = {
    'aywb': 'annapurna yojana', 'lbs-wb': 'lokkhir bhandar', 'kbs': 'krishok bondhu', 'wbkanyashree': 'kanyashree',
    'wbrupashree': 'rupashree', 'wbsaboojsathi': 'sabooj sathi cycle', 'oapwb': 'old age pension', 'wpwb': 'bidhoba bhata',
    'jjoaps': 'jai johar pension', 'obs': 'taposili bondhu', 'bay': 'banglar awas', 'yuvasree': 'yuvasree bhata',
    'wbsccs': 'student credit card', 'pm-kisan': 'pm kisan', 'pmjdy': 'jan dhan account', 'pmuy': 'ujjwala gas',
    'pmmvy': 'matri vandana', 'apy': 'atal pension', 'pmsby': 'suraksha bima', 'pmjjby': 'jeevan jyoti bima'}
BANGLISH_Q = {'amount': ['{b} e koto taka pai?', '{b} er taka koto?'], 'docs': ['{b} er jonno ki ki kagoj lagbe?', '{b} document list bolo'],
              'where': ['{b} kothay apply korbo?', '{b} form kothay pabo?'], 'who': ['{b} ke ke pabe?', '{b} eligibility ki?']}

PERSONAS = {
    'aywb': lambda r: dict(gender='f', age=r.choice([22, 28, 35, 41, 49, 55, 59, 63, 67]), govt_job=r.random() < .12, taxpayer=r.random() < .08),
    'kbs': lambda r: dict(gender=r.choice('mf'), age=r.randint(24, 70), job=r.choice(['farmer'] * 5 + ['labour']), land_acres=r.choice([0, .3, .5, 1, 1.5, 2])),
    'pm-kisan': lambda r: dict(gender='m', age=r.randint(25, 70), job=r.choice(['farmer'] * 4 + ['labour']), land_acres=r.choice([0, .4, 1, 2]), taxpayer=r.random() < .1),
    'wbkanyashree': lambda r: dict(gender='f', age=r.randint(12, 20), student_class=r.choice([7, 8, 9, 10, 11, 12, None]), married=r.random() < .1, income_year=r.choice([60000, 90000, 110000, 150000, 200000])),
    'wbrupashree': lambda r: dict(gender='f', age=r.randint(17, 26), married=r.random() < .08, income_year=r.choice([70000, 100000, 140000, 180000])),
    'wbsaboojsathi': lambda r: dict(gender=r.choice('mf'), age=r.randint(13, 18), student_class=r.choice([8, 9, 10, 11, 12]), school_type=r.choice(['govt', 'govt', 'aided', 'private', 'madrasah'])),
    'oapwb': lambda r: dict(gender=r.choice('mf'), age=r.choice([55, 58, 61, 66, 72, 80]), caste=r.choice(['gen', 'obc']), other_pension=r.random() < .15),
    'wpwb': lambda r: dict(gender='f', age=r.randint(35, 80), widow=r.random() < .85, other_pension=r.random() < .1),
    'jjoaps': lambda r: dict(gender=r.choice('mf'), age=r.choice([56, 60, 64, 71]), caste=r.choice(['st', 'st', 'sc', 'gen']), other_pension=r.random() < .1),
    'obs': lambda r: dict(gender=r.choice('mf'), age=r.choice([57, 60, 65, 73]), caste=r.choice(['sc', 'sc', 'st', 'obc']), other_pension=r.random() < .1),
    'yuvasree': lambda r: dict(gender=r.choice('mf'), age=r.randint(17, 48), job=r.choice(['unemployed'] * 3 + ['student', 'labour']), education=r.choice([6, 8, 10, 12, 15])),
    'pmuy': lambda r: dict(gender=r.choice('ffm'), age=r.randint(17, 60), has_lpg=r.random() < .2),
    'pmmvy': lambda r: dict(gender='f', age=r.randint(17, 34), pregnant=True, first_child=r.random() < .75, govt_job=r.random() < .07),
    'apy': lambda r: dict(gender=r.choice('mf'), age=r.randint(16, 48), taxpayer=r.random() < .1),
    'pmjjby': lambda r: dict(gender=r.choice('mf'), age=r.randint(16, 60)),
    'pmsby': lambda r: dict(gender=r.choice('mf'), age=r.randint(16, 76)),
    'bay': lambda r: dict(gender=r.choice('mf'), age=r.randint(25, 70), kutcha_house=r.random() < .7),
}


def persona_text(slug, p, r):
    name = r.choice(WOMEN if p.get('gender') == 'f' else MEN)
    d = r.choice(DISTRICTS)
    bits = [f"আমি {name}, {d} জেলায় থাকি।" if r.random() < .5 else f"আমার নাম {name}, বাড়ি {d}।"]
    if 'age' in p:
        bits.append(f"বয়স {bn(p['age'])} বছর।")
    if p.get('caste') in CASTE_BN:
        bits.append(f"আমরা {CASTE_BN[p['caste']]}।")
    if p.get('job') == 'farmer':
        bits.append(f"চাষ করি, নিজের নামে {bn(p.get('land_acres', 0))} একর জমি।" if p.get('land_acres') else 'চাষ করি, কিন্তু জমি নিজের নামে নেই।')
    elif p.get('job') == 'labour':
        bits.append('দিনমজুরি করি।')
    elif p.get('job') == 'unemployed':
        bits.append(f"কাজ নেই, {bn(p.get('education', 10))} ক্লাস পর্যন্ত পড়েছি।")
    if p.get('student_class'):
        st = {'govt': 'সরকারি স্কুলে', 'aided': 'সরকার-পোষিত স্কুলে', 'private': 'প্রাইভেট স্কুলে', 'madrasah': 'মাদ্রাসায়'}.get(p.get('school_type'), '')
        bits.append(f"{st} {bn(p['student_class'])} ক্লাসে পড়ি।".strip())
    elif slug == 'wbkanyashree':
        bits.append('এখন পড়াশোনা বন্ধ।')
    if p.get('married') is not None:
        bits.append('বিয়ে হয়ে গেছে।' if p['married'] else 'এখনও বিয়ে হয়নি।')
    if p.get('widow') is not None:
        bits.append('স্বামী মারা গেছেন।' if p['widow'] else 'স্বামী আছেন।')
    if p.get('income_year'):
        bits.append(f"পরিবারের বছরে আয় মোটামুটি {inr(p['income_year'])} টাকা।")
    if p.get('govt_job'):
        bits.append('সরকারি স্কুলে স্থায়ী চাকরি করি।' if slug != 'pmmvy' else 'আমি সরকারি চাকরি করি।')
    if p.get('taxpayer'):
        bits.append('আয়কর দিই।')
    if p.get('other_pension'):
        bits.append('এখন একটা অন্য সরকারি পেনশন পাই।')
    if p.get('has_lpg') is not None:
        bits.append('বাড়িতে আগে থেকেই গ্যাস আছে।' if p['has_lpg'] else 'বাড়িতে গ্যাস নেই, কাঠে রান্না করি।')
    if p.get('pregnant'):
        bits.append('আমি গর্ভবতী, ' + ('এটা প্রথম সন্তান।' if p.get('first_child') else 'আগে একটা সন্তান আছে।'))
    if p.get('kutcha_house') is not None:
        bits.append('মাটির কাঁচা বাড়িতে থাকি।' if p['kutcha_house'] else 'পাকা বাড়ি আছে।')
    ask = r.choice(['আমি কি {n} পাব?', '{n}-এর জন্য আমি কি যোগ্য?', 'আমার কি {n}-এ আবেদন করা উচিত?'])
    return ' '.join(bits) + ' ' + ask.format(n=CARDS[slug]['name'])


def build(seed, n_persona_per_scheme, n_problem, out_path, test=False):
    r = random.Random(seed)
    rows = []
    names = [c['name'] for c in CARDS.values()]
    def add(prompt, completion, kind, slug, **gold):
        rows.append(dict(prompt=fix_gen(prompt, names), completion=fix_gen(completion, names), kind=kind, scheme=slug, **gold))
    for slug, c in CARDS.items():
        n = c['name']
        for kind, fn in (('overview', ans_overview), ('amount', ans_amount), ('docs', ans_docs), ('where', ans_where), ('who', ans_who)):
            for t in (Q_TEST if test else Q)[kind]:
                add(t.format(n=n), fn(c, r), kind, slug)
            bq = BANGLISH_Q_TEST if test else BANGLISH_Q
            if slug in BANGLISH and kind in bq:
                for t in bq[kind]:
                    add(t.format(b=BANGLISH[slug]), fn(c, r), kind + '_banglish', slug)
        if slug in PERSONAS:
            for _ in range(n_persona_per_scheme):
                p = PERSONAS[slug](r)
                ans, verdict = ans_persona(slug, c, p, r)
                add(persona_text(slug, p, r), ans, 'persona', slug, verdict=verdict)
    for _ in range(n_problem):
        slug = r.choice(list(CARDS))
        qs, a = r.choice(PROBLEMS)
        add(r.choice(qs).format(s=CARDS[slug]['name']), a + '\n\n' + r.choice(CLOSE[2:]), 'problem', slug)
    # out-of-scope / honesty rows
    for q in ['মঙ্গল গ্রহে জমি কেনার সরকারি প্রকল্প আছে?', 'আগামী মাসে ভাতা কত বাড়বে?', 'কোন প্রকল্পে মাসে ২০,০০০ টাকা দেয়?',
              'অন্য রাজ্যের রেশন কার্ড দিয়ে এখানে কী পাব?'][(2 if test else 0):]:
        add(q, 'এ বিষয়ে আমার কাছে নিশ্চিত সরকারি তথ্য নেই, তাই আন্দাজে বলব না। আপনার BDO অফিস, দুয়ারে সরকার ক্যাম্প '
               'বা সংশ্লিষ্ট দপ্তরের সরকারি পোর্টালে জিজ্ঞেস করুন। আমি পশ্চিমবঙ্গের পরিচিত প্রকল্পগুলোর শর্ত, কাগজপত্র আর '
               'আবেদনের পদ্ধতি বুঝিয়ে দিতে পারি।', 'unknown', '')
    seen, uniq = set(), []
    for x in rows:                       # one row per distinct question
        if x['prompt'] not in seen:
            seen.add(x['prompt']); uniq.append(x)
    rows = uniq
    r.shuffle(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for x in rows:
            f.write(json.dumps(dict(x, system=SYSTEM), ensure_ascii=False) + '\n')
    return rows


if __name__ == '__main__':
    tr = build(seed=11, n_persona_per_scheme=45, n_problem=260, out_path=os.path.join(ROOT, 'data', 'seed_train.jsonl'))
    te = build(seed=99, n_persona_per_scheme=6, n_problem=30, out_path=os.path.join(ROOT, 'data', 'test_heldout.jsonl'), test=True)
    tp = {x['prompt'] for x in tr}
    te2 = [x for x in te if x['prompt'] not in tp]          # held-out must not repeat a training prompt
    with open(os.path.join(ROOT, 'data', 'test_heldout.jsonl'), 'w', encoding='utf-8') as f:
        for x in te2:
            f.write(json.dumps(dict(x, system=SYSTEM), ensure_ascii=False) + '\n')
    from collections import Counter
    st = [f'train rows {len(tr)}  unique prompts {len(tp)}', f'test rows {len(te2)} (removed {len(te) - len(te2)} overlapping)',
          'train kinds ' + str(Counter(x["kind"] for x in tr)), 'test kinds ' + str(Counter(x["kind"] for x in te2)),
          'persona verdicts train ' + str(Counter(x.get("verdict") for x in tr if x['kind'] == 'persona')),
          'avg answer chars ' + str(round(sum(len(x["completion"]) for x in tr) / len(tr)))]
    open(os.path.join(ROOT, 'data', 'stats.txt'), 'w', encoding='utf-8').write('\n'.join(st))
    print('\n'.join(st))

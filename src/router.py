"""Sahayak Bangla router: "the rules decide, the model explains".

Eligibility questions ("I am X, age N, ... will I get scheme S?") are the model's weak spot (round-2 model told 50 of
61 eligible people "hard to get"). Here a small parser reads the person's facts from the Bengali / Banglish question,
src/rules.py decides yes/no with the official conditions, and the answer is built from the checked fact cards.
If a fact the rules need is missing, the assistant ASKS for it instead of guessing.
Every other question goes to the fine-tuned model.

route(question) -> dict(path='rules'|'ask'|'model', scheme, persona, verdict, answer)
"""
import json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_seed as B
from rules import check

CARDS = B.CARDS
NAMES = [c['name'] for c in CARDS.values()]
D = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')

ALIAS = {   # Bengali + Banglish ways people name each scheme
    'aywb': ['অন্নপূর্ণা', 'annapurna'], 'lbs-wb': ['লক্ষ্মীর ভান্ডার', 'লক্ষ্মীর ভাণ্ডার', 'lokkhir bhandar', 'lakshmir bhandar'],
    'kbs': ['কৃষক বন্ধু', 'কৃষকবন্ধু', 'krishok bondhu', 'krishak bandhu'], 'wbkanyashree': ['কন্যাশ্রী', 'kanyashree', 'kanyasree'],
    'wbrupashree': ['রূপশ্রী', 'rupashree', 'rupasree'], 'wbsaboojsathi': ['সবুজ সাথী', 'সবুজসাথী', 'sabooj sathi', 'sabuj sathi'],
    'oapwb': ['বার্ধক্য ভাতা', 'বার্ধক্য পেনশন', 'old age pension', 'bardhakya bhata'],
    'wpwb': ['বিধবা ভাতা', 'বিধবা পেনশন', 'bidhoba bhata', 'widow pension'], 'jjoaps': ['জয় জোহার', 'jai johar', 'joy johar'],
    'obs': ['তপশিলি বন্ধু', 'তফসিলি বন্ধু', 'তপসিলি বন্ধু', 'taposili bondhu', 'taposili bandhu'],
    'bay': ['বাংলার আবাস', 'banglar awas', 'bangla awas'], 'yuvasree': ['যুবশ্রী', 'yuvasree', 'yubashree'],
    'wbsccs': ['স্টুডেন্ট ক্রেডিট কার্ড', 'student credit card'], 'pm-kisan': ['কিষাণ', 'কিসান', 'pm kisan', 'pm kishan', 'pm-kisan'],
    'pmjdy': ['জন ধন', 'জনধন', 'jan dhan', 'jandhan'], 'pmuy': ['উজ্জ্বলা', 'ujjwala'], 'pmmvy': ['মাতৃ বন্দনা', 'মাতৃবন্দনা', 'matri vandana', 'matru vandana'],
    'apy': ['অটল পেনশন', 'atal pension'], 'pmsby': ['সুরক্ষা বিমা', 'সুরক্ষা বীমা', 'suraksha bima'],
    'pmjjby': ['জীবন জ্যোতি', 'jeevan jyoti', 'jivan jyoti'],
}
# facts each scheme's rules cannot do without (otherwise we ask)
NEED = {'aywb': ['gender', 'age'], 'lbs-wb': ['gender', 'age'], 'kbs': ['job'], 'pm-kisan': ['job'],
        'wbkanyashree': ['age', 'student_class'], 'wbrupashree': ['age', 'married'], 'wbsaboojsathi': ['student_class', 'school_type'],
        'oapwb': ['age'], 'wpwb': ['widow'], 'jjoaps': ['caste', 'age'], 'obs': ['caste', 'age'], 'bay': ['kutcha_house'],
        'yuvasree': ['age', 'job'], 'pmuy': ['gender', 'has_lpg'], 'pmmvy': ['pregnant', 'age'], 'apy': ['age'],
        'pmsby': ['age'], 'pmjjby': ['age'], 'wbsccs': ['age']}
ASK = {'gender': 'আপনি কি মহিলা, না পুরুষ?', 'age': 'আপনার বয়স কত বছর?', 'job': 'আপনি কি নিজে চাষ করেন? নিজের নামে জমি আছে কি?',
       'student_class': 'কোন ক্লাসে পড়েন?', 'school_type': 'সরকারি, সরকার-পোষিত, মাদ্রাসা না প্রাইভেট স্কুল?',
       'married': 'এখনও বিয়ে হয়নি তো? এটা কি প্রথম বিয়ে?', 'widow': 'স্বামী কি মারা গেছেন?',
       'caste': 'আপনারা কি তফসিলি জাতি (SC), তফসিলি উপজাতি (ST), OBC না সাধারণ?', 'kutcha_house': 'এখন কি কাঁচা / ভাঙা বাড়িতে থাকেন, না পাকা বাড়ি?',
       'has_lpg': 'বাড়িতে আগে থেকে গ্যাস (LPG) সংযোগ আছে কি?', 'pregnant': 'আপনি কি এখন গর্ভবতী?'}
ASK_WORDS = re.compile(r'পাব|পাবো|পাবেন|যোগ্য|উচিত|পেতে পারি|পাওয়া যাবে|পাবে কি|pabo|joggo|pete pari|apply korte pari|pabe')
SELF = re.compile(r'আমি|আমার|আমাদের|আমরা|ami |amar |amra ')


def num(s):
    return int(s.translate(D).replace(',', ''))


def find_scheme(q):
    ql = q.lower()
    for s, c in sorted(CARDS.items(), key=lambda kv: -len(kv[1]['name'])):   # full official name wins
        if c['name'].lower() in ql:
            return s
    best = None   # else the longest alias found (so 'তপশিলি বন্ধু' beats the shorter 'বার্ধক্য ভাতা')
    for s, al in ALIAS.items():
        for a in al:
            if a.lower() in ql and (best is None or len(a) > best[1]):
                best = (s, len(a))
    return best[0] if best else None


def strip_names(q):
    for n in sorted(NAMES, key=len, reverse=True):
        q = q.replace(n, ' ')
    for al in ALIAS.values():
        for a in al:
            q = re.sub(re.escape(a), ' ', q, flags=re.I)
    return q


def parse(q):
    t = strip_names(q)
    tl = t.lower()
    p = {}
    m = re.search(r'বয়স\s*([০-৯\d]+)|([০-৯\d]+)\s*বছর(?:ের)?(?!\s*ধরে)|(?:age|boyos|bayos)\s*(\d+)|(\d+)\s*(?:bochor|bochhor|years?)', t, re.I)
    if m: p['age'] = num(next(g for g in m.groups() if g))
    # gender
    if re.search(r'মহিলা|মেয়ে|স্বামী|গর্ভবতী|বিধবা|গৃহবধূ|\bmohila\b|\bmeye\b|swami|widow', tl): p['gender'] = 'f'
    elif re.search(r'পুরুষ|ছেলে|স্ত্রী|\bchhele\b|\bpurush\b', tl): p['gender'] = 'm'
    else:
        if any(n in t for n in B.WOMEN): p['gender'] = 'f'
        elif any(n in t for n in B.MEN): p['gender'] = 'm'
    # caste (ST before SC: 'উপজাতি' contains 'জাতি')
    if re.search(r'উপজাতি|\bST\b|আদিবাসী|adivasi', t): p['caste'] = 'st'
    elif re.search(r'তফসিলি জাতি|তপশিলি জাতি|\bSC\b', t): p['caste'] = 'sc'
    elif re.search(r'\bOBC\b|ওবিসি', t): p['caste'] = 'obc'
    elif re.search(r'সাধারণ|general', tl): p['caste'] = 'gen'
    # work / land
    if re.search(r'চাষ করি|চাষি|কৃষক|chash kori|chasi|farmer', tl):
        p['job'] = 'farmer'
        m = re.search(r'([০-৯\d]+(?:[.\.][০-৯\d]+)?)\s*(?:একর|acre)', t, re.I)
        if m: p['land_acres'] = float(m.group(1).translate(D))
        if re.search(r'জমি (?:নিজের নামে )?নেই|নিজের নামে নেই|জমি নেই|ভাগে চাষ|অন্যের জমি|jomi nei', tl): p['land_acres'] = 0
        elif 'land_acres' not in p and re.search(r'নিজের নামে .*জমি|নিজের জমি', tl): p['land_acres'] = 1
    elif re.search(r'দিনমজুর|মজুরি করি|কাজ করি|dinmojur|labour', tl): p['job'] = 'labour'
    elif re.search(r'কাজ নেই|বেকার|চাকরি নেই|bekar|kaj nei|unemployed', tl): p['job'] = 'unemployed'
    elif re.search(r'কলেজে পড়ি|বিশ্ববিদ্যালয়|ইউনিভার্সিটি|কোচিং|উচ্চশিক্ষা|college e pori', tl): p['job'] = 'student'
    m = re.search(r'([০-৯\d]+)\s*(?:ক্লাস|শ্রেণি|শ্রেণী|class)\s*পর্যন্ত', t)
    if m: p['education'] = num(m.group(1))
    m = re.search(r'([০-৯\d]+)\s*(?:ক্লাসে|শ্রেণিতে|শ্রেণীতে|class e)\s*পড়', t) or re.search(r'class\s*(\d+)\s*e\s*pori', tl)
    if m: p['student_class'] = num(m.group(1))
    if re.search(r'সরকার-পোষিত|সরকার পোষিত|এইডেড|aided', tl): p['school_type'] = 'aided'
    elif re.search(r'মাদ্রাসা|madrasa', tl): p['school_type'] = 'madrasah'
    elif re.search(r'প্রাইভেট|বেসরকারি|private', tl): p['school_type'] = 'private'
    elif re.search(r'সরকারি স্কুলে\s*(?:[০-৯\d]+\s*)?(?:ক্লাসে|শ্রেণিতে)|সরকারি স্কুলে পড়|govt school', tl): p['school_type'] = 'govt'
    # family
    if re.search(r'বিয়ে হয়নি|অবিবাহিত|বিয়ে হবে|biye hoyni', tl): p['married'] = False
    elif re.search(r'বিয়ে হয়ে গেছে|বিবাহিত|বিয়ে হয়েছে|biye hoye geche', tl): p['married'] = True
    if re.search(r'স্বামী[^।?]{0,25}মারা|বিধবা|স্বামী নেই|widow|swami mara', tl): p['widow'] = True
    elif re.search(r'স্বামী আছেন|স্বামী বেঁচে', tl): p['widow'] = False
    m = re.search(r'আয়[^।]*?([০-৯\d][০-৯\d,]*)\s*টাকা', t)
    if m: p['income_year'] = num(m.group(1)) * (12 if re.search(r'মাসে', m.group(0)) else 1)
    if re.search(r'সরকারি চাকরি করি|স্থায়ী চাকরি করি|সরকারি স্কুলে স্থায়ী চাকরি|সরকারি কর্মচারী|sarkari chakri kori', tl): p['govt_job'] = True
    if re.search(r'(?:আয়কর|ইনকাম ট্যাক্স|ট্যাক্স) দিই না|tax dii na', tl): p['taxpayer'] = False
    elif re.search(r'আয়কর দিই|ইনকাম ট্যাক্স দিই|ট্যাক্স দিই|tax dii', tl): p['taxpayer'] = True
    if re.search(r'(?:অন্য|আরেকটা|একটা অন্য).*পেনশন পাই|পেনশন পাই', tl): p['other_pension'] = True
    if re.search(r'গ্যাস নেই|গ্যাস সংযোগ নেই|কাঠে রান্না|কয়লায় রান্না|gas nei', tl): p['has_lpg'] = False
    elif re.search(r'গ্যাস আছে|গ্যাস সংযোগ আছে|gas ache', tl): p['has_lpg'] = True
    if re.search(r'গর্ভবতী|pregnant|পেটে বাচ্চা', tl):
        p['pregnant'] = True
        p['first_child'] = not re.search(r'আগে (?:একটা|একটি|দুটো)? ?সন্তান আছে|দ্বিতীয় সন্তান|আরেকটা সন্তান|ditiyo', tl)
    if re.search(r'কাঁচা বাড়ি|মাটির বাড়ি|মাটির কাঁচা|ভাঙা বাড়ি|বাড়ি নেই|গৃহহীন|kacha bari', tl): p['kutcha_house'] = True
    elif re.search(r'পাকা বাড়ি|paka bari', tl): p['kutcha_house'] = False
    return p


def route(q, model_answer=None, seed=0):
    slug = find_scheme(q)
    is_persona = bool(slug) and bool(ASK_WORDS.search(q.lower())) and bool(SELF.search(q.lower())) and slug in NEED
    if not is_persona:
        return dict(path='model', scheme=slug, persona=None, verdict=None, answer=model_answer)
    p = parse(q)
    missing = [k for k in NEED[slug] if k not in p]
    card = CARDS[slug]
    if missing:
        qs = ' '.join(ASK[k] for k in missing[:2])
        ans = (f"{card['name']}-এর শর্ত মিলছে কিনা ঠিকভাবে বলতে আরও একটু জানা দরকার: {qs}\n\n"
               f"এগুলো জানালে আমি সরকারি শর্ত মিলিয়ে বলে দেব।")
        return dict(path='ask', scheme=slug, persona=p, verdict='ask', answer=B.fix_gen(ans, NAMES))
    ans, verdict = B.ans_persona(slug, card, p, random.Random(seed))
    return dict(path='rules', scheme=slug, persona=p, verdict=verdict, answer=B.fix_gen(ans, NAMES))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    for q in sys.argv[1:] or ['আমি রুমা, বয়স ৩৫ বছর। আমার কি অন্নপূর্ণা যোজনায় আবেদন করা উচিত?']:
        r = route(q); print(r['path'], r['verdict'], r['persona']); print(r['answer'])

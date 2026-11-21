"""Machine-checkable eligibility rules per scheme (from the official cards) + persona model.

A persona = dict(age, gender 'f'/'m', caste 'gen'/'sc'/'st'/'obc', job, land_acres, student_class, school_type,
married, widow, govt_job, taxpayer, income_year, has_lpg, pregnant, first_child, kutcha_house).
check(slug, persona) -> (verdict, reasons) where verdict in {'yes','no','maybe'}.
'maybe' = persona passes the hard rules we can check, but some conditions need office verification.
"""

def _age(p, lo=None, hi=None):
    a = p.get('age')
    if a is None:
        return None
    return (lo is None or a >= lo) and (hi is None or a <= hi)


def check(slug, p):
    r_ok, r_no = [], []
    def need(cond, ok_txt, no_txt):
        if cond is None:
            return
        (r_ok if cond else r_no).append(ok_txt if cond else no_txt)

    if slug in ('aywb', 'lbs-wb'):
        need(p.get('gender') == 'f', 'আপনি মহিলা', 'এই প্রকল্প শুধু মহিলাদের জন্য')
        need(_age(p, 25, 60), f"বয়স {p.get('age')} — ২৫ থেকে ৬০-এর মধ্যে", f"বয়স {p.get('age')} — শর্ত ২৫ থেকে ৬০ বছর")
        need(not p.get('govt_job'), 'স্থায়ী সরকারি চাকরি নেই', 'স্থায়ী সরকারি চাকরি থাকলে পাওয়া যায় না')
        need(not p.get('taxpayer'), 'আয়কর দেন না', 'আয়কর দিলে পাওয়া যায় না')
    elif slug == 'kbs':
        need(p.get('job') == 'farmer', 'আপনি চাষ করেন', 'নিজে চাষ করতে হবে')
        need((p.get('land_acres') or 0) > 0, 'নিজের নামে চাষের জমি আছে', 'নিজের নামে (বা নথিভুক্ত বর্গা) চাষের জমি লাগবে')
    elif slug == 'pm-kisan':
        need(p.get('job') == 'farmer' and (p.get('land_acres') or 0) > 0, 'নিজের নামে চাষের জমি আছে', 'জমির মালিক কৃষক পরিবার হতে হবে')
        need(not p.get('taxpayer'), 'আয়কর দেন না', 'আয়কর দাতারা বাদ')
        need(not p.get('govt_job'), 'উচ্চপদস্থ সরকারি চাকরি নেই', 'উচ্চপদস্থ সরকারি কর্মীরা বাদ')
    elif slug == 'wbkanyashree':
        need(p.get('gender') == 'f', 'মেয়ে', 'শুধু মেয়েদের জন্য')
        need(_age(p, 13, 19), f"বয়স {p.get('age')}", f"বয়স {p.get('age')} — K1-এর জন্য ১৩–১৮, K2-এর জন্য ১৮–১৯")
        need(p.get('student_class') is not None and p.get('student_class') >= 8, 'পড়াশোনা চলছে (অষ্টম শ্রেণি বা উপরে)', 'স্কুল/কলেজে নিয়মিত পড়তে হবে (অষ্টম শ্রেণি বা উপরে)')
        need(not p.get('married'), 'অবিবাহিত', 'বিবাহিত হলে পাওয়া যায় না')
        need(p.get('income_year') is None or p['income_year'] <= 120000, 'পারিবারিক আয় ১,২০,০০০-এর মধ্যে', 'পারিবারিক আয় বছরে ১,২০,০০০ টাকার বেশি')
    elif slug == 'wbrupashree':
        need(p.get('gender') == 'f', 'মেয়ে', 'বিয়ের পাত্রী (মেয়ে) নিজে আবেদন করেন')
        need(_age(p, 18), f"বয়স {p.get('age')} — ১৮ বা তার বেশি", 'মেয়ের বয়স অন্তত ১৮ হতে হবে')
        need(not p.get('married'), 'এখনও অবিবাহিত, এটি প্রথম বিয়ে', 'প্রথম বিয়ের ক্ষেত্রেই পাওয়া যায়')
        need(p.get('income_year') is None or p['income_year'] <= 150000, 'পারিবারিক আয় ১,৫০,০০০-এর মধ্যে', 'পারিবারিক আয় বছরে ১,৫০,০০০ টাকার বেশি')
    elif slug == 'wbsaboojsathi':
        need(p.get('student_class') in (9, 10, 11, 12), f"{p.get('student_class')} শ্রেণিতে পড়ে", 'নবম থেকে দ্বাদশ শ্রেণির ছাত্রছাত্রীদের জন্য')
        need(p.get('school_type') in ('govt', 'aided', 'madrasah'), 'সরকারি/সরকার-পোষিত স্কুল', 'বেসরকারি স্কুলের ছাত্রছাত্রীরা পায় না')
    elif slug == 'oapwb':
        need(_age(p, 60), f"বয়স {p.get('age')} — ৬০ বা তার বেশি", f"বয়স {p.get('age')} — ৬০ বছর হতে হবে (প্রতিবন্ধীদের ৫৫)")
        need(not p.get('other_pension'), 'অন্য সরকারি পেনশন নেই', 'অন্য সরকারি পেনশন পেলে একসঙ্গে পাওয়া যায় না')
        need(p.get('caste') not in ('sc', 'st') or None, 'সাধারণ / OBC — রাজ্যের বার্ধক্য ভাতা দেখুন', None)
    elif slug == 'wpwb':
        need(bool(p.get('widow')), 'বিধবা', 'শুধু বিধবা মহিলাদের জন্য')
        need(not p.get('other_pension'), 'অন্য সরকারি পেনশন নেই', 'অন্য সরকারি পেনশন পেলে পাওয়া যায় না')
    elif slug == 'jjoaps':
        need(p.get('caste') == 'st', 'তফসিলি উপজাতি (ST)', 'শুধু তফসিলি উপজাতিদের (ST) জন্য')
        need(_age(p, 60), f"বয়স {p.get('age')} — ৬০ বা তার বেশি", 'বয়স ৬০ হতে হবে')
        need(not p.get('other_pension'), 'অন্য পেনশন নেই', 'অন্য পেনশন পেলে পাওয়া যায় না')
    elif slug == 'obs':
        need(p.get('caste') == 'sc', 'তফসিলি জাতি (SC)', 'শুধু তফসিলি জাতিদের (SC) জন্য')
        need(_age(p, 60), f"বয়স {p.get('age')} — ৬০ বা তার বেশি", 'বয়স ৬০ হতে হবে')
        need(not p.get('other_pension'), 'অন্য পেনশন নেই', 'অন্য পেনশন পেলে পাওয়া যায় না')
    elif slug == 'bay':
        need(bool(p.get('kutcha_house')), 'কাঁচা/ভাঙা বাড়িতে থাকেন', 'গৃহহীন বা কাঁচা/ভাঙা বাড়ির পরিবারের জন্য')
    elif slug == 'yuvasree':
        need(_age(p, 18, 45), f"বয়স {p.get('age')} — ১৮ থেকে ৪৫", 'বয়স ১৮ থেকে ৪৫ হতে হবে')
        need(p.get('job') == 'unemployed', 'বেকার', 'বেকার হতে হবে')
        need(p.get('education', 0) >= 8, 'অন্তত অষ্টম শ্রেণি পাস', 'অন্তত অষ্টম শ্রেণি পাস হতে হবে')
    elif slug == 'pmuy':
        need(p.get('gender') == 'f' and _age(p, 18), 'প্রাপ্তবয়স্ক মহিলা', 'বাড়ির প্রাপ্তবয়স্ক মহিলাকেই আবেদন করতে হবে')
        need(not p.get('has_lpg'), 'বাড়িতে গ্যাস সংযোগ নেই', 'বাড়িতে আগে থেকে LPG সংযোগ থাকলে পাওয়া যায় না')
    elif slug == 'pmmvy':
        need(p.get('gender') == 'f' and bool(p.get('pregnant')), 'গর্ভবতী', 'গর্ভবতী মায়েদের জন্য')
        need(_age(p, 19), f"বয়স {p.get('age')} — অন্তত ১৯", 'বয়স অন্তত ১৯ হতে হবে')
        need(p.get('first_child', True), 'প্রথম সন্তান', 'সাধারণত প্রথম সন্তানের জন্য')
        need(not p.get('govt_job'), 'সরকারি চাকরি নেই', 'সরকারি চাকরিতে মাতৃত্বকালীন সুবিধা পেলে পাওয়া যায় না')
    elif slug == 'apy':
        need(_age(p, 18, 40), f"বয়স {p.get('age')} — ১৮ থেকে ৪০", 'যোগ দেওয়ার বয়স ১৮ থেকে ৪০')
        need(not p.get('taxpayer'), 'আয়কর দেন না', 'আয়কর দাতারা যোগ দিতে পারেন না')
    elif slug == 'pmsby':
        need(_age(p, 18, 70), f"বয়স {p.get('age')} — ১৮ থেকে ৭০", 'বয়স ১৮ থেকে ৭০ হতে হবে')
    elif slug == 'pmjjby':
        need(_age(p, 18, 50), f"বয়স {p.get('age')} — ১৮ থেকে ৫০", 'বয়স ১৮ থেকে ৫০ হতে হবে')
    elif slug == 'wbsccs':
        need(_age(p, None, 40), f"বয়স {p.get('age')} — ৪০-এর মধ্যে", 'বয়স ৪০-এর বেশি হলে পাওয়া যায় না')
        need(p.get('job') == 'student', 'পড়াশোনা চলছে', 'উচ্চশিক্ষা বা কোচিংয়ে ভর্তি থাকতে হবে')
    elif slug == 'pmjdy':
        need(True, 'যেকোনো ভারতীয় নাগরিক খুলতে পারেন', None)
    verdict = 'no' if r_no else ('maybe' if r_ok else 'maybe')
    if not r_no and r_ok:
        verdict = 'yes'
    return verdict, r_ok, r_no

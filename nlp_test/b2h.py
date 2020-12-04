import hgtk

CHO = {
    '[0,0,0,1,0,0]':'ㄱ',
    '[1,0,0,1,0,0]':'ㄴ',
    '[0,1,0,1,0,0]':'ㄷ',
    '[0,0,0,0,1,0]':'ㄹ',
    '[1,0,0,0,1,0]':'ㅁ',
    '[0,0,0,1,1,0]':'ㅂ',
    '[0,0,0,0,0,1]':'ㅅ',
    '[1,1,0,1,1,0]':'ㅇ',
    '[0,0,0,1,0,1]':'ㅈ',
    '[0,0,0,0,1,1]':'ㅊ',
    '[1,1,0,1,0,0]':'ㅋ',
    '[1,1,0,0,1,0]':'ㅌ',
    '[1,0,0,1,1,0]':'ㅍ',
    '[0,1,0,1,1,0]':'ㅎ',

    '[0,0,0,0,0,1],[0,0,0,1,0,0]':'ㄲ',
    '[0,0,0,0,0,1],[0,1,0,1,0,0]':'ㄸ',
    '[0,0,0,0,0,1],[0,0,0,1,1,0]':'ㅃ',
    '[0,0,0,0,0,1],[0,0,0,0,0,1]':'ㅆ',
    '[0,0,0,0,0,1],[0,0,0,1,0,1]':'ㅉ',
}
JOONG = {
    '[1,1,0,0,0,1]':'ㅏ',
    '[0,0,1,1,1,0]':'ㅑ',
    '[0,1,1,1,0,0]':'ㅓ',
    '[1,0,0,0,1,1]':'ㅕ',
    '[1,0,1,0,0,1]':'ㅗ',
    '[0,0,1,1,0,1]':'ㅛ',
    '[1,0,1,1,0,0]':'ㅜ',
    '[1,0,0,1,0,1]':'ㅠ',
    '[0,1,0,1,0,1]':'ㅡ',
    '[1,0,1,0,1,0]':'ㅣ',
    '[1,1,1,0,1,0]':'ㅐ',
    '[1,0,1,1,1,0]':'ㅔ',
    '[0,0,1,1,1,0],[1,1,1,0,1,0]':'ㅒ',
    '[0,0,1,1,0,0]':'ㅖ',
    '[1,1,1,0,0,1]':'ㅘ',
    '[1,1,1,0,0,1],[1,1,1,0,1,0]':'ㅙ',
    '[1,0,1,1,1,1]':'ㅚ',
    '[1,1,1,1,0,0]':'ㅝ',
    '[1,1,1,1,0,0],[1,1,1,0,1,0]':'ㅞ',
    '[1,0,1,1,0,0],[1,1,1,0,1,0]':'ㅟ',
    '[0,1,0,1,1,1]':'ㅢ',
}
JONG = {
    '[1,0,0,0,0,0]':'ㄱ',
    '[0,1,0,0,1,0]':'ㄴ',
    '[0,0,1,0,1,0]':'ㄷ',
    '[0,1,0,0,0,0]':'ㄹ',
    '[0,1,0,0,0,1]':'ㅁ',
    '[1,1,0,0,0,0]':'ㅂ',
    '[0,0,1,0,0,0]':'ㅅ',
    '[0,1,1,0,1,1]':'ㅇ',
    '[1,0,1,0,0,0]':'ㅈ',
    '[0,1,1,0,0,0]':'ㅊ',
    '[0,1,1,0,1,0]':'ㅋ',
    '[0,1,1,0,0,1]':'ㅌ',
    '[0,1,0,0,1,1]':'ㅍ',
    '[0,0,1,0,1,1]':'ㅎ',

    '[1,0,0,0,0,0],[1,0,0,0,0,0]':'ㄲ',
    '[1,0,0,0,0,0],[0,0,1,0,0,0]':'ㄳ',
    '[0,1,0,0,1,0],[1,0,1,0,0,0]':'ㄵ',
    '[0,1,0,0,1,0],[0,0,1,0,1,1]':'ㄶ',
    '[0,1,0,0,0,0],[1,0,0,0,0,0]':'ㄺ',
    '[0,1,0,0,0,0],[0,1,0,0,0,1]':'ㄻ',
    '[0,1,0,0,0,0],[1,1,0,0,0,0]':'ㄼ',
    '[0,1,0,0,0,0],[0,0,1,0,0,0]':'ㄽ',
    '[0,1,0,0,0,0],[0,1,1,0,0,1]':'ㄾ',
    '[0,1,0,0,0,0],[0,1,0,0,1,1]':'ㄿ',
    '[0,1,0,0,0,0],[0,0,1,0,1,1]':'ㅀ',
    '[1,1,0,0,0,0],[0,0,1,0,0,0]':'ㅄ',
    '[0,0,1,1,0,0]':'ㅆ',
}
MIN = {
    '[1,1,0,1,0,1]':[[0,0,0,1,0,0],[1,1,0,0,0,1]],      # '가',
    #'[1,0,0,1,0,0]':[[0,0,0,1,0,0],[1,1,0,0,0,1]],     # '나',
    #'[0,1,0,1,0,0]': '다',
    '[0,1,0,0,0,0],[1,1,0,0,0,1]': [[0,0,0,0,1,0],[1,1,0,0,0,1]],      # '라',
    #'[1,0,0,0,1,0]': '마',
    #'[0,0,0,1,1,0]': '바',
    '[1,1,1,0,0,0]': [[0,0,0,0,0,1],[1,1,0,0,0,1]],      # '사',
    #'[1,1,0,0,0,1]': [[1,1,0,1,1,0],[1,1,0,0,0,1]],      # '아',
    #'[0,0,0,1,0,1]': '자',
    '[0,1,1,0,0,0],[1,1,0,0,0,1]': [[0,0,0,0,1,1],[1,1,0,0,0,1]],      # '차',
    #'[1,1,0,1,0,0]': '카',
    #'[1,1,0,0,1,0]': '타',
    #'[1,0,0,1,1,0]': '파',
    #'[0,1,0,1,1,0]': '하',

    '[0,0,1,0,0,1]': '_',

    '[0,0,0,1,1,1],[1,1,0,0,0,1]':[[0,0,0,1,0,0],[0,1,1,1,0,0],[0,0,1,0,0,0]],  #것
    '[1,0,0,1,1,1]':[[0,1,1,1,0,0],[1,0,0,0,0,0]],      #'억',
    '[0,1,1,1,1,1]':[[0,1,1,1,0,0],[0,1,0,0,1,0]],      #'언',
    '[0,1,1,1,1,0]':[[0,1,1,1,0,0],[0,1,0,0,0,0]],      #얼
    '[1,0,0,0,0,1]':[[1,0,0,0,1,1],[0,1,0,0,1,0]],    #'연'
    '[1,1,0,0,1,1]':[[1,0,0,0,1,1],[0,1,0,0,0,0]],  #열
    '[1,1,0,1,1,1]':[[1,0,0,0,1,1],[0,1,1,0,1,1]],  #'영'
    '[1,0,1,1,0,1]':[[1,0,1,0,0,1],[1,0,0,0,0,0]],  #'옥',
    '[1,1,1,0,1,1]':[[1,0,1,0,0,1],[0,1,0,0,1,0]],      #'온',
    '[1,1,1,1,1,1]':[[1,0,1,0,0,1],[0,1,1,0,1,1]],        #'옹',
    '[1,1,0,1,1,0]':[[1,0,1,1,0,0],[0,1,0,0,1,0]],      #'운',
    '[1,1,1,1,0,1]':[[1,0,1,1,0,0],[0,1,0,0,0,0]],      #'울',
    '[1,0,1,0,1,1]':[[0,1,0,1,0,1],[0,1,0,0,1,0]],      #은
    '[0,1,1,1,0,1]':[[0,1,0,1,0,1],[0,1,0,0,0,0]],      #'을',
    '[1,1,1,1,1,0]':[[1,0,1,0,1,0],[0,1,0,0,1,0]],      #'인',
    '[1,0,0,0,0,0],[0,1,1,1,0,0]':[[0,0,0,1,0,0],[0,1,0,1,0,1],[0,0,0,0,1,0],[1,1,1,0,1,0],[0,0,0,0,0,1],[0,1,1,1,0,0]],    #그래서
    '[1,0,0,0,0,0],[1,0,0,1,0,0]':[[0,0,0,1,0,0],[0,1,0,1,0,1],[0,0,0,0,1,0],[0,1,1,1,0,0],[1,0,0,1,0,0],[1,1,0,0,0,1]],      #'그러나',
    '[1,0,0,0,0,0],[0,1,0,0,1,0]':[[0,0,0,1,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,1,1,1,0,0],[1,0,0,0,1,0],[1,0,0,0,1,1],[1,0,0,1,0,0]],  #'그러면',
    '[1,0,0,0,0,0],[0,1,0,0,0,1]':[[0,0,0,1,0,0],[0,1,0,1,0,1],[0,0,0,0,1,0],[0,1,1,1,0,0],[1,0,0,0,1,0],[0,1,0,1,0,1],[0,0,0,0,1,0],[1,0,1,0,0,1]],  #'그러므로',
    '[1,0,0,0,0,0],[1,0,1,1,1,0]':[[0,0,0,1,0,0], [0,1,0,1,0,1], [0,0,0,0,1,0], [0,1,1,1,0,0], [1,0,0,1,0,0], [0,1,0,1,0,0], [1,0,1,1,1,0]],    #'그런데',
    '[1,0,0,0,0,0],[1,0,1,0,0,1]':[[0,0,0,1,0,0], [0,1,0,1,0,1], [0,0,0,0,1,0], [1,0,1,0,1,0], [0,0,0,1,0,0], [1,0,1,0,0,1]],   #'그리고',
    '[1,0,0,0,0,0],[1,0,0,0,1,1]':[[0,0,0,1,0,0], [0,1,0,1,0,1], [0,0,0,0,1,0], [1,0,1,0,1,0], [0,1,0,1,1,0], [1,1,0,0,0,1], [1,0,0,0,1,1]],    #'그리하여',
}
NUM = {
    #'[0,0,1,1,1,1]':'수표',
    '[1,0,0,0,0,0]':'1',
    '[1,1,0,0,0,0]':'2',
    '[1,0,0,1,0,0]':'3',
    '[1,0,0,1,1,0]':'4',
    '[1,0,0,0,1,0]':'5',
    '[1,1,0,1,0,0]':'6',
    '[1,1,0,1,1,0]':'7',
    '[1,1,0,0,1,0]':'8',
    '[0,1,0,1,0,0]':'9',
    '[0,1,0,1,1,0]':'0',

    '[0,0,0,0,0,0]':' ',
    '[0,1,0,0,1,1]':'.',
    '[0,0,0,0,1,0]':',',
    '[0,1,0,0,1,0]':'-',
    '[0,1,1,0,0,1]':'?',
    '[0,0,1,0,0,1]':'_',
    '[0,1,1,0,1,0]':'!',
    '[0,0,0,0,0,1],[0,0,0,0,0,1],[0,0,0,0,0,1]':'……',
    '[0,0,0,0,1,0],[0,1,0,0,0,0]':':',
    '[0,0,0,0,1,1],[0,1,1,0,0,0]':';'
}
PUNC = {
    '[0,0,0,0,0,0]':' ',
    '[0,1,0,0,1,1]':'.',
    #'[0,1,0,0,1,0]':'-',
    '[0,1,1,0,0,1]':'?',
    '[0,1,1,0,1,0]':'!',
    '[0,0,0,0,0,1],[0,0,0,0,0,1],[0,0,0,0,0,1]':'……',
    '[0,0,0,0,1,0],[0,1,0,0,0,0]':':',
    '[0,0,0,0,1,1],[0,1,1,0,0,0]':';',
    '[0,0,0,0,1,0],[0,0,0,0,0,0]':', ',
    '[0,0,0,1,1,1],[0,0,1,1,0,0]':'/'
}
P_PUNC = {
    '[0,0,0,0,0,1],[0,1,1,0,0,1]':[[0,0,1,0,1,1],[0,0,1,0,0,0]],
    '[0,0,0,0,1,0],[0,1,1,0,0,1]':[[0,0,1,0,1,1],[0,1,0,0,0,0]],
    '[0,0,0,0,1,1],[0,1,1,0,0,1]':[[0,0,1,0,1,1],[0,1,1,0,0,0]],
    '[0,1,1,0,0,1],[0,1,1,0,0,0]':[[0,0,0,0,1,1],[0,0,1,0,1,1]],
    '[0,1,1,0,0,1],[0,0,1,0,0,0]':[[0,0,0,0,0,1],[0,0,1,0,1,1]],
#    '[0,0,0,0,0,1],[0,0,1,0,0,1]':[[0,0,1,0,0,1],[0,0,1,0,0,0]], #밑줄
    '[0,1,1,0,0,1]': [0,0,1,0,1,1],
}
F_PUNC = {
    '[0,0,0,0,0,1],[0,1,1,0,0,1]':'"',
    '[0,0,0,0,1,0],[0,1,1,0,0,1]':'<',
    '[0,0,0,0,1,1],[0,1,1,0,0,1]':'《',
    '[0,1,1,0,0,1],[0,1,1,0,0,0]':'(',  #[
    '[0,1,1,0,0,1],[0,0,1,0,0,0]':'(',
    '[0,1,1,0,0,1]': '"'
}
B_PUNC = {
    '[0,0,1,0,1,1],[0,0,1,0,0,0]':'"',
    '[0,0,1,0,1,1],[0,1,0,0,0,0]':'>',
    '[0,0,1,0,1,1],[0,1,1,0,0,0]':'》',
    '[0,0,0,0,1,1],[0,0,1,0,1,1]':")",  #]
    '[0,0,0,0,0,1],[0,0,1,0,1,1]':')',
    '[0,0,1,0,1,1]':'"'
}


def b2hcvt(aaa):
    # 앞뒤 2 punc
    a1, a2 = 0,0
    aa00, aa1, aap= '','',''
    for i in range(len(aaa)):
        aa = aaa[i]
        aa0 = str(aa00).replace(" ", "")+","+str(aa).replace(" ", "")
        if aa0 in F_PUNC:
            a1 = i
            aap = str(P_PUNC[aa0]).replace(" ","").replace("[[","[").replace("]]","]")
            aa1 = F_PUNC[aa0]
        elif aa0 in B_PUNC and aa0 == aap:
            a2 = i
            aa2 = B_PUNC[aa0]
            aaa[a1] = list(aa1)
            aaa[a2] = list(aa2)
            aaa[a1-1] = ""
            aaa[a2-1] = ""
            a1, a2 = 0,0
            aap = ''
        aa00 = aa

    #앞뒤 1 punc
    a1, a2 = 0, 0
    aa00, aa1, aap = '', '', ''
    for i in range(len(aaa)):
        aa = aaa[i]
        aa0 = str(aa).replace(" ", "")
        if aa0 in F_PUNC:
            a1 = i
            aap = str(P_PUNC[aa0]).replace(" ", "")
            aa1 = F_PUNC[aa0]
        if aa0 == aap:
            try:
                a2 = i
                aa2 = B_PUNC[aa0]
                aaa[a1] = list(aa1)
                aaa[a2] = list(aa2)
                a1, a2 = 0, 0
            except:
                break

    #띄어쓰기 기준으로 단어 분해, punc
    b = []
    a1=''
    r = []
    for a2 in aaa:
        r.append(a2)
        a = str(a1).replace(" ", "") + ',' + str(a2).replace(" ", "")
        if a in PUNC:
            try:
                r = r[0:-2]
                r.append(PUNC[a])
                r.append("")
                b.append(r)
                r = []
                a1 = a2
                continue
            except:
                pass
        a = str(a2).replace(" ","")
        if a in PUNC:
            r = r[0:-1]
            r.append(PUNC[a])
            b.append(r)
            r = []
            continue

        a1 = a2
    if r != []:
        b.append(r)
    aaa = b

    #숫자, punc
    supyo = [0,0,1,1,1,1]
    b = []
    for aa in aaa:
        try:
            if aa[0] == supyo:
                stop = 0
                aa.remove(supyo)
                r = []
                for a in aa:
                    try:
                        if stop == 0:
                            a = str(a).replace(" ","")
                            r.append(NUM[a])
                        elif stop == 1:
                            r.append(a)
                    except:
                        r.append(a)
                        stop = 1
                b.append(r)
                continue
            b.append(aa)
        except:
            pass
    aaa = b

    #약어
    b = []
    for aa in aaa:
        a1=''
        bb = []
        for a2 in aa:
            bb.append(a2)
            a = str(a2).replace(" ","")
            if a in MIN:
                bb.remove(a2)
                bb.extend(i for i in list(MIN[a]))

            a = str(a1).replace(" ","")+','+str(a2).replace(" ","")
            if a in MIN:
                bb.remove(a1)
                bb.remove(a2)
                bb.extend(i for i in list(MIN[a]))
            a1 = a2
        b.append(bb)
    aaa = b

    b = []
    for aa in aaa:
        w = []
        s = []
        checker = 0
        cho, joong, jong = '', '', ''
        for a in aa:
            a2 = str(a).replace(" ","")
            if a2 in CHO:
                if checker == 1:
                    # if cho == CHO[a2]:
                    #     s[-1] = CHO[str(a2).replace(" ","")+","+str(a2).replace(" ","")]
                    #     continue
                    if cho == 'ㅅ':
                        s[-1] = hgtk.text.CHO_COMP[cho][CHO[a2]]
                        continue
                    else:
                        s.append('ㅏ')
                if checker > 0:
                    w.append(s)
                    s = []
                    cho, joong, jong = '', '', ''
                s = []
                cho = CHO[a2]
                s.append(cho)
                checker = 1
            elif a2 in JOONG:
                if checker == 2 and JOONG[a2] == 'ㅖ':
                    s.append('ㅆ')
                    checker = 3
                    cho, joong, jong = '', '', ''
                    continue
                else:
                    joong = JOONG[a2]
                if checker == 2 and JOONG[a2] == 'ㅐ':
                    s[-1] = hgtk.text.JOONG_COMP[s[-1]][JOONG[a2]]
                    checker = 2
                    cho, joong, jong = '', '', ''
                elif checker != 1:
                    w.append(s)
                    s = ['ㅇ', joong]
                    checker = 2
                    cho, joong, jong = '', '', ''
                elif checker == 1:
                    s.append(joong)
                    checker = 2
                    cho, joong, jong = '', '', ''
            elif a2 in JONG:
                if checker == 1:
                    s.append('ㅏ')
                if checker < 3:
                    jong = JONG[a2]
                    s.append(jong)
                    w.append(s)
                    checker = 3
                    s = []
                    cho, joong = '', ''
                elif checker == 3:
                    try:
                        jong = hgtk.text.JONG_COMP[jong][JONG[a2]]
                        w[-1] = w[-1][0:-1]
                        w[-1].append(jong)
                        s = []
                        chekcer = 4
                    except:
                        pass
            else:
                if checker == 1:
                    s.append('ㅏ')
                    checker = 2
                w.append(s)
                w.append(a2)
                s = []
                checker = 5
                cho, joong, jong = '', '', ''
        if checker == 1:
            s.append('ㅏ')
            w.append(s)
            checker = 2
        elif checker == 2:
            w.append(s)
            checker = 2
        b.append(w)

    aaa = b

    result = ''
    for aa in aaa:
        for a in aa:
            result += hgtk.text.compose2(a)
            if a == '':
                result += ' '
    result = result.replace("  "," ")
    return result



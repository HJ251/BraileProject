B_CODE = {
    '[0,0,0,1,0,0]':chr(10248),
    '[0,1,1,1,0,0]':chr(10254),
    '[1,0,0,0,1,1]':chr(10289),
    '[1,0,1,0,0,1]':chr(10277),
    '[0,0,1,1,0,1]':chr(10284),
    '[1,0,0,1,0,1]':chr(10281),
    '[0,1,0,1,0,1]':chr(10282),
    '[1,0,1,0,1,0]':chr(10261),
    '[1,1,1,0,1,0]':chr(10263),
    '[1,0,1,1,1,0]':chr(10269),
    '[0,0,1,1,1,0]':chr(10268),
    '[1,1,1,0,0,1]':chr(10279),
    '[1,0,1,1,1,1]':chr(10301),
    '[1,1,1,1,0,0]':chr(10255),
    '[1,0,1,1,0,0]':chr(10253),
    '[0,1,0,1,1,1]':chr(10298),
    '[0,0,1,0,1,0]':chr(10260),
    '[0,1,0,0,0,1]':chr(10274),
    '[0,0,1,0,0,0]':chr(10244),
    '[0,1,1,0,1,1]':chr(10294),
    '[1,0,1,0,0,0]':chr(10245),
    '[0,0,1,1,0,0]':chr(10252),
    '[1,1,0,1,0,1]':chr(10283),
    '[0,1,0,0,0,0]':chr(10242),
    '[0,0,0,1,1,0]':chr(10264),
    '[1,1,1,0,0,0]':chr(10247),
    '[1,1,0,0,0,1]':chr(10275),
    '[0,0,0,1,0,1]':chr(10280),
    '[0,1,1,0,0,0]':chr(10246),
    '[1,0,0,1,1,1]':chr(10297),
    '[0,1,1,1,1,1]':chr(10302),
    '[0,1,1,1,1,0]':chr(10270),
    '[1,0,0,0,0,1]':chr(10273),
    '[1,1,0,0,1,1]':chr(10291),
    '[1,1,0,1,1,1]':chr(10299),
    '[1,0,1,1,0,1]':chr(10285),
    '[1,1,1,0,1,1]':chr(10295),
    '[1,1,1,1,1,1]':chr(10303),
    '[1,1,1,1,0,1]':chr(10287),
    '[1,0,1,0,1,1]':chr(10293),
    '[0,1,1,1,0,1]':chr(10286),
    '[1,1,1,1,1,0]':chr(10271),
    '[0,0,1,1,1,1]':chr(10300),
    '[1,0,0,0,0,0]':chr(10241),
    '[1,1,0,0,0,0]':chr(10243),
    '[1,0,0,1,0,0]':chr(10249),
    '[1,0,0,1,1,0]':chr(10265),
    '[1,0,0,0,1,0]':chr(10257),
    '[1,1,0,1,0,0]':chr(10251),
    '[1,1,0,1,1,0]':chr(10267),
    '[1,1,0,0,1,0]':chr(10259),
    '[0,1,0,1,0,0]':chr(10250),
    '[0,1,0,1,1,0]':chr(10266),
    '[0,0,1,0,0,1]':chr(10276),
    '[0,0,0,0,0,0]':chr(10240),
    '[0,1,0,0,1,1]':chr(10290),
    '[0,1,0,0,1,0]':chr(10258),
    '[0,1,1,0,1,0]':chr(10262),
    '[0,0,0,1,1,1]':chr(10296),
    '[0,0,0,0,1,0]':chr(10256),
    '[0,1,1,0,0,1]':chr(10278),
    '[0,0,0,0,1,1]':chr(10288),
    '[0,0,0,0,0,1]':chr(10272),
    '[0,0,1,0,1,1]':chr(10292)
}

def b2ccvt(a):
    aa = ''
    for b in a:
        aa += B_CODE[str(b).replace(" ","")]
    return aa



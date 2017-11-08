import fileinput
from collections import OrderedDict
import collections

givenstr = ""
for line in fileinput.input():
    givenstr = givenstr + line


def fib_dyn(givenstr):
    startyear = givenstr[0].split(",")[0].split("-")[0]
    startmonth = givenstr[0].split(",")[0].split("-")[1]
    endyear = givenstr[0].split(",")[1].split("-")[0].replace(" ", "")
    endmonth = givenstr[0].split(",")[1].split("-")[1]
    capture = {}
    givenstr = givenstr[2:len(givenstr) - 1]

    for line in range(len(givenstr)):
        year = givenstr[line].split("-")[0]
        month = givenstr[line].split("-")[1]
        if (len(givenstr[line].split(",")) > 2):
            data = givenstr[line].split(",")
        if (int(year + month) >= int(startyear + startmonth) and int(year + month) <
            int(endyear + endmonth)):
            datatype = data[1].replace(" ", "")
            dataofdatatype = data[2].replace(" ", "")
            key = str(year + "-" + month)
            if key not in capture:
                capture[key] = {}
                capture[str(key)][datatype] = {}
                capture[str(key)][datatype] = dataofdatatype
            else:
                if datatype in capture[str(key)]:
                    capture[str(key)][datatype] = str(int(capture[str(key)][datatype])+ int(dataofdatatype))
                else:
                    capture[str(key)][datatype] = {}
                    capture[str(key)][datatype] = dataofdatatype
            if (int(dataofdatatype) <= 0):
                capture[str(key)].pop(datatype)
            capture = collections.OrderedDict(sorted(capture.items(), reverse=True))
    for keyH, valueH in capture.items():
        capture[keyH] = OrderedDict(sorted(capture[keyH].items(), key=lambda t: t[0]))
        print(keyH, end="")
        for key, value in capture[keyH].items():
            print(", " + key + ", " + value, end="")
    print("")


givenstr = givenstr.split("\n")
fib_dyn(givenstr)

import random

def hoeffding_calculation():
    print("Computing hoeffding values!")
    no_of_coins=1000
    no_of_tosses=10
    epochs=1000
    v1_avg=[0] * epochs
    vrand_avg=[0] * epochs
    vmin_avg=[0] * epochs
    c1=1
    for l in range(epochs):
        count = [0] * 1000
        crand = random.randint(0, 999)
        print("Epoch no: ", l, " Crand: ", crand)
        cmin = 1
        for n in range(no_of_coins):
            for m in range(no_of_tosses):
                toss=random.randint(1,2)
                #print("toss result: ", toss)
                if toss == 1:
                    count[n]=count[n]+0.1
            if count[n] < count[cmin]:
                cmin = n
        print("v1: ", count[1])
        print("vrand: ", count[crand])
        print("vmin: ", count[cmin])
        v1_avg[l] = count[1]
        vrand_avg[l] = count[crand]
        vmin_avg[l] = count[cmin]
    print("v1_avg: ", sum(v1_avg) / float(len(v1_avg)))
    print("vrand_avg: ", sum(vrand_avg) / float(len(vrand_avg)))
    print("vmin_avg: ", sum(vmin_avg) / float(len(vmin_avg)))

hoeffding_calculation()

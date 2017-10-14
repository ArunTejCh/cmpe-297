class Person:
    pplCount = 0

    def __init__(self, name, age):
        self.name = name
        self.age = age
        Person.pplCount += 1

    def displayTotal(self):
        print "People count %d" % Person.pplCount

    def showPerson(self):
        print "Name : ", self.name, ", Age: ", self.age


person1 = Person("Arun", 27)
person2 = Person("Tej", 34)

person1.showPerson()
person2.showPerson()

print "Total no of people %d" % Person.pplCount
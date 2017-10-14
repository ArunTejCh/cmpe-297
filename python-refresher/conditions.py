# Say a greeting depending upon time of the day and language
def say_greeting(lang, time):
    if lang == "english":
        if time == "morning":
            return "Good Morning!"
        elif time == "evening":
            return "Good Evening!"
        elif time == "night":
            return "Good Night!"
        else:
            return "Hello!"
    elif lang == "french":
        if time == "morning":
            return "Bonjour!"
        elif time == "evening":
            return "Bonsoir!"
        elif time == "night":
            return "Bonne nuit!"
        else:
            return "Bonjour!"


print say_greeting("english", "morning")
print say_greeting("french", "evening")
def add(a, b):
    return a + b


print(add(1, 2))  # 3
print(add(*(1, 2)))  # 3


def add_any(*args):
    return sum(args)


print(add_any(1, 2, 3))  # 6
print(add_any(1, 2, 3, 4))  # 10


def keywords(name="Shane", course="575k"):
    return f"{name} is teaching {course}"


print(keywords(name="Agatha"))
print(keywords(**{"name": "Agatha"}))


def keywords_any(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

keywords_any(name="Shane", course="575k"))
keywords_any(name="Shane", course="575k", foo="bar"))
keywords_any(**{"name": "Shane", "course": "575k"}))
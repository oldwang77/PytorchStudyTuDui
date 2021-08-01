class Person:
    def __call__(self, name):
        print("__call__" + "Hello" + name)

    def hello(self, name):
        print("hello" + name)


person = Person()

# call可以直接通过这样的方式调用执行
person("zhangsan")
person.hello("lisi")

# __call__Hellozhangsan
# hellolisi
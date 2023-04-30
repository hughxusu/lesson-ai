# 面向对象的其他特性

## 类属性

类属性就是类的所有实例对象所共有属性，类属性可以使用类对象或实例对象访问。

```python
class Animal(object):
    leg_num = 0

class Horse(Animal):
    leg_num = 4
    
animal = Animal()
print(animal)

one = Horse()
two = Horse()

print(one.leg_num)
print(two.leg_num)
print(Horse.leg_num)
```

类属性只能通过类对象修改，不能通过实例对象修改，如果通过实例对象修改类属性，表示的是创建了一个实例属性。

```python
Horse.leg_num = 6
print(one.leg_num)
print(two.leg_num)
print(Horse.leg_num)

# 创建一个实例属性
one.leg_num = 8
print(one.leg_num)
print(two.leg_num)
print(Horse.leg_num)
```

## 类方法

类属性就是类的所有实例对象所共有方法，类属性可以使用类对象或实例对象访问；类方法一般和类属性配合使用。

```python
class Horse():
    __leg_num = 4

    @classmethod
    def get_leg_num(cls):
        return cls.__leg_num

    @classmethod
    def set_leg_num(cls, num):
        cls.__leg_num = num

print(Horse.get_leg_num())
Horse.set_leg_num(6)
print(Horse.get_leg_num())
```

## 静态方法

需要通过装饰器 `@staticmethod` 来进行修饰，与实例和类对象均无关，可以认为是绑定在类上的单独函数。

```python
class Horse():
    @staticmethod
    def desc():
        print('这是一个马类用于创建实例对象...')

horse = Horse()
Horse.desc()
horse.desc()
```


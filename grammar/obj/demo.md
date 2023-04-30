# 案例：面向对象版学员管理系统

## 系统需求

> [!tip]
>
> 使用面向对象编程思想，实现一个用户管理系统，能够完成如下功能：
>
> 1. 添加用户
> 2. 删除用户
> 3. 修改用户名称
> 4. 查询用户名称
> 5. 显示所有用户信息
> 6. 保存学员信
> 7. 退出系统
>
> 该系统有功能提示并能接收用户输入，且学员数据存储在文件中，系统启动后数据可以从文件中读取。

### 项目所需功能

1. `__dict__`

返回类内部所有属性和方法对应的字典

```python
class A(object):
    a = 0

    def __init__(self):
        self.b = 1


aa = A()
# 返回类内部所有属性和方法对应的字典
print(A.__dict__)
# 返回实例属性和值组成的字典
print(aa.__dict__)
```

2. `eval`

`eval()` 函数用来执行一个字符串表达式，并返回表达式的值。

## 代码实现

### 定义用户对象

```python
class User(object):
    def __init__(self, phone, name):
        self.phone = phone
        self.name = name

    def __str__(self):
        return f"phone: {self.phone}, name: {self.name}"

    def is_this(self, phone):
        if self.phone == phone:
            return True
        return False
```

### 定义系统对象

```python
from user import User

class UserSystem(object):

    def __init__(self):
        self.users = []

    @staticmethod
    def print_hint():
        print('=' * 20)
        print('请选择系统功能:')
        print('1. 添加用户')
        print('2. 删除用户')
        print('3. 修改用户信息')
        print('4. 查询用户信息')
        print('5. 显示所有用户信息')
        print('6. 保存学员信息')
        print('7. 退出系统')
        print('-' * 20)

    def run(self):
        self.load_users()

        while True:
            self.print_hint()
            in_str = input('请输入您需要的功能序号:')

            if in_str.isdigit():
                choice = int(in_str)
                print(choice)

                if choice == 1:
                    self.add_user()
                elif choice == 2:
                    self.del_user()
                elif choice == 3:
                    self.edit_user()
                elif choice == 4:
                    self.search_user()
                elif choice == 5:
                    self.print_all()
                elif choice == 6:
                    self.save_user()
                elif choice == 7:
                    break
                else:
                    print('输入错误请重新输入')
            else:
                print('输入错误请重新输入')

    def find_user(self, phone):
        for user in self.users:
            if user.is_this(phone):
                return user

        return False

    def add_user(self):
        phone = input('请输入手机号:')
        name = input('请输入用户名:')

        if self.find_user(phone):
            print('该手机号已存在')
        else:
            user = User(phone, name)
            self.users.append(user)
            print('添加用户成功')

    def del_user(self):
        phone = input('请输入手机号:')

        user = self.find_user(phone)
        if user:
            self.users.remove(user)
            print('删除用户成功')
        else:
            print('该用户不存在')

    def edit_user(self):
        phone = input('请输入手机号:')

        user = self.find_user(phone)
        if user:
            name = input('请输入用户名:')
            user.name = name
            print(user)
            print('修改用户成功')
        else:
            print('该用户不存在')

    def search_user(self):
        phone = input('请输入手机号:')

        user = self.find_user(phone)
        if user:
            print(user)
        else:
            print('该用户不存在')

    def save_user(self):
        f = open('users.data', 'w')
        users_list = [i.__dict__ for i in self.users]
        f.write(str(users_list))
        f.close()

    def load_users(self):
        try:
            f = open('users.data', 'r')
        except:
            f = open('users.data', 'w')
        else:
            data = f.read()
            users_list = eval(data)
            self.users = [User(i['phone'], i['name']) for i in users_list]
        finally:
            f.close()

    def print_all(self):
        for user in self.users:
            print(user)
```

### 调用系统对象

```python
from system import UserSystem

if __name__ == '__main__':
    system = UserSystem()
    system.run()
```


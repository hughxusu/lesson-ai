# 用户管理

设计一个简易的用户管理系统。

软件的生命周期

```mermaid
graph LR
    需求分析-->概要设计-->详细设计-->编码和单元测试-->综合测试-->软件维护
```

## 系统需求

> [!tip]
>
> 实现一个用户管理系统，能够完成如下功能：
>
> 1. 添加用户
> 2. 删除用户
> 3. 修改用户名称
> 4. 查询用户名称
> 5. 显示所有用户信息
> 6. 退出系统
>
> 该系统有功能提示并能接收用户输入。

### 显示系统功能

进入系统后，打印如下文字：

```shell
请选择系统功能:
1. 添加用户
2. 删除用户
3. 修改用户信息
4. 查询用户信息
5. 显示所有用户信息
6. 退出系统

请输入您需要的功能序号:
```

当用户输入的数字不再 1~6 之间提示 `输入错误请重新输入`

### 添加用户

用户信息包括：用户名和手机号

进入添加用户后提示

```shell
请输入手机号：
请输入用户名：
```

如果手机号已存在，提示 `该手机号已存在` 并返回显示系统功能。如果手机号不存在，添加用户，并提示 `添加用户成功`，并退回选择系统。

### 删除用户

进入删除用户后提示 `请输入手机号:` 如果手机号不存在提示 `该用户不存在` ，如果手机号存在，删除用户，并提示 `删除用户成功`，并退回选择系统。

### 修改用户

进入修改用户后提示 `请输入手机号:` 如果手机号不存在提示 `该用户不存在` ，如果手机号存在，提示 `请输入新用户名`，修改用户名，打印用户信息并提示用户名修改成功，打印新的手机号和用户名。并退回选择系统。

### 查询用户

进入查询用户后提示 `请输入手机号:` 如果手机号不存在提示 `该用户不存在` ，如果手机号存在，打印新的手机号和用户名。并退回选择系统。

### 显示所有用户信息

变量并打印所有用户信息。并退回选择系统。

信息格式，每一行

```shell
phone: 131123456, name: Tom
```

## 代码实现

```python
users = []


def print_hint():
    print('='*20)
    print('请选择系统功能:')
    print('1. 添加用户')
    print('2. 删除用户')
    print('3. 修改用户信息')
    print('4. 查询用户信息')
    print('5. 显示所有用户信息')
    print('6. 退出系统')
    print('-' * 20)


def find_user(phone):
    global users

    for user in users:
        if user['phone'] == phone:
            return user

    return False


def print_user(user):
    print(f"phone: {user['phone']}, name: {user['name']}")


def add_user():
    phone = input('请输入手机号:')
    name = input('请输入用户名:')
    global users

    if find_user(phone):
        print('该手机号已存在')

    user = {'phone': phone, 'name': name}
    users.append(user)
    print('添加用户成功')


def del_user():
    phone = input('请输入手机号:')
    global users

    user = find_user(phone)
    if user:
        users.remove(user)
        print('删除用户成功')
    else:
        print('该用户不存在')


def edit_user():
    phone = input('请输入手机号:')
    global users

    user = find_user(phone)
    if user:
        name = input('请输入用户名:')
        user['name'] = name
        print_user(user)
        print('修改用户成功')
    else:
        print('该用户不存在')


def search_user():
    phone = input('请输入手机号:')
    global users

    user = find_user(phone)
    if users:
        print_user(user)
    else:
        print('该用户不存在')


def print_all():
    for user in users:
        print_user(user)


while True:
    print_hint()

    in_str = input('请输入您需要的功能序号:')
    if in_str.isdigit():
        choice = int(in_str)
        if choice == 1:
            add_user()
        elif choice == 2:
            del_user()
        elif choice == 3:
            edit_user()
        elif choice == 4:
            search_user()
        elif choice == 5:
            print_all()
        elif choice == 6:
            break
        else:
            print('输入错误请重新输入')
    else:
        print('输入错误请重新输入')
```


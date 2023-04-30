users = []

class User():
    def __init__(self, phone, name):
        self.phone = phone
        self.name = name

    def __str__(self):
        return f"phone: {self.phone}, name: {self.name}"

    def is_this(self, phone):
        if self.phone == phone:
            return True
        return False

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
        if user.is_this(phone):
            return user

    return False

def print_user(user):
    print(user)

def add_user():
    phone = input('请输入手机号:')
    name = input('请输入用户名:')
    global users

    if find_user(phone):
        print('该手机号已存在')

    user = User(phone, name)
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
        user.name = name
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

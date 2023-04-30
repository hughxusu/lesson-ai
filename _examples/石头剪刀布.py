import random

# 从控制台输入要出的拳 —— 石头（1）／剪刀（2）／布（3）
player = int(input("请出拳 石头（1）／剪刀（2）／布（3）："))

computer = random.randint(1, 3)

print('=' * 50)
print('player: %d, computer: %d' % (player, computer))
print('=' * 50)
if ((player == 1 and computer == 2) or
        (player == 2 and computer == 3) or
        (player == 3 and computer == 1)):

    print("玩家获胜！！")
elif player == computer:
    print("平局！！！")
else:
    print("电脑获胜！！")

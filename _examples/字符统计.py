docs = '''
黄河安澜是中华儿女的千年期盼。
近年来，我走遍了黄河上中下游9省区。
无论是黄河长江“母亲河”，还是碧波荡漾的青海湖、逶迤磅礴的雅鲁藏布江；
无论是南水北调的世纪工程，还是塞罕坝林场的“绿色地图”；
无论是云南大象北上南归，还是藏羚羊繁衍迁徙……这些都昭示着，人不负青山，青山定不负人。
'''

signs = ['。', '；', '“', '”', '，', '……', '、', '9', '\n']

result = docs
for sign in signs:
    result = result.replace(sign, '')

chars = set(result)
print(f'字符数量为: {len(chars)}')

max_num = 0
max_char = ''
for char in chars:
    num = result.count(char)
    print(f'字符： {char}, 使用了: {num} 次')
    if num > max_num:
        max_char = char
        max_num = num

print(f'使用最多的字符是: {max_char}, 次数是: {max_num}')

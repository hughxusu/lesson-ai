src_name = input('请输入文件名: ')

index = src_name.rfind('.')

if index > 0:
    suffix = src_name[index:]

dst_name = src_name[:index] + '_copy' + suffix

src = open(src_name, 'rb')
dst = open(dst_name, 'wb')

while True:
    buff = src.read(1024)
    if len(buff) == 0:
        break
    dst.write(buff)

src.close()
dst.close()

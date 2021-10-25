import socket
import struct
address = ("127.0.0.1", 9050)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(address)
s.listen(1000)


client, addr = s.accept()
print('got connected from', addr)

buf = ''
while len(buf)<4:
    buf += client.recv(4-len(buf)).decode()
size = struct.unpack('!i', buf)
print("receiving %s bytes" % size)

with open('tst2.png', 'wb') as img:
    while True:
        data = client.recv(1024)
        if not data:
            break
        img.write(data)
print('received, yay!')

client.close()
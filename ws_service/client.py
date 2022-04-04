# -*- coding: utf-8 -*-
# +
import json
from websocket import create_connection
import base64

uri = "ws://0.0.0.0:1234"
text = 'Привет всем моим друзьям'

sample_path = 'test.wav'
data = {"text": text}

ws = create_connection(uri)
ws.send(json.dumps(data))
response = json.loads(ws.recv())
audio = base64.b64decode(response['audio'])

with open(sample_path,'wb') as f:
    f.write(audio)
    
if response["event"] == "success":
    print('\n')
    print('-'*7)
    print('Success')
    print('-'*7)
    print('\n')
elif response["event"] == "error":
    print('\n')
    print('Error!\n{}'.format(response["msg"]))
    print('\n')

ws.close()
# -


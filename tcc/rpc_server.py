from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler
import datetime
import time
import subprocess
import infer
import os

# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

#Config
porta = 8081
endereco = "192.168.0.14"

# Create server
server = SimpleXMLRPCServer((endereco, porta),
                            requestHandler=RequestHandler)
server.register_introspection_functions()
print("Servidor registrado na porta: "+endereco+":"+str(porta))

# SOMA - EXEMPLO
def exemplo(x,y):
    return x + y
server.register_function(exemplo)

# Salvar foto
def salvarFoto(nome, imagem64):
    pasta = "../treinamento/originais/"+nome+"/"
    if not os.path.exists(pasta):
        os.makedirs(pasta)
    print("Imagem recebida. Iniciando escrita em: "+pasta)
    name = pasta+str(len(os.listdir(pasta)))+".png"
    with open(name, "wb") as f:
        print("Imagem recebida. Iniciando escrita em: "+pasta)    
        f.write(imagem64.decode('base64'))      

    return '{ "result": "Imagem de '+nome+' salva com sucesso." }'
server.register_function(salvarFoto)

# Treinar
def treinar():
    gerarClassifier()
    with open("../treinamento/feature/classifier.pkl", "rb") as f:
        class1 = f.read().encode('base64')      

    return '{ "classifier":"'+class1+'" }'
server.register_function(treinar)

def descobrir(classifier64, imagem64):
    print("Imagem recebida. Iniciando escrita...")
    with open("../temp/imagem_temporaria.png", "wb") as f:
	    f.write(imagem64.decode('base64'))
    print("Escrita concluida.")
    
    print("Classificador recebido. Iniciando escrita...")
    with open("../temp/classifier.pkl", "wb") as f:
        f.write(classifier64.decode('base64'))
    print("Escrita concluida.")
    
    result = infer.infer("../temp/classifier.pkl", "../temp/imagem_temporaria.png", True)
    print("Imagem Processada"+ result)
    return result
server.register_function(descobrir)

def gerarClassifier():
    workdir = "/root/openface/"
    
    print("Iniciando treinamento")
    start = time.time()
    #raw_input("alinhamento")
    print(datetime.datetime.now().time())
    bashCommand = "./util/align-dlib.py ./treinamento/originais/ align outerEyesAndNose ./treinamento/alinhadas/ --size 96"
    p = subprocess.Popen(bashCommand, shell=True, cwd=workdir)
    p.wait()
    print(datetime.datetime.now().time())
    end = time.time()
    print("levou: "+str(end-start))

    #raw_input("representacao")
    print(datetime.datetime.now().time())         
    start = time.time()
    bashCommand = "./batch-represent/main.lua -outDir ./treinamento/feature/ -data ./treinamento/alinhadas/"
    p = subprocess.Popen(bashCommand, shell=True, cwd=workdir)
    p.wait()
    print(datetime.datetime.now().time())
    end = time.time()
    print("levou: "+str(end-start))

    #raw_input("classificador")
    print(datetime.datetime.now().time())
    start = time.time()
    bashCommand = "./demos/classifier.py train ./treinamento/feature/"
    p = subprocess.Popen(bashCommand, shell=True, cwd=workdir)
    p.wait()
    print(datetime.datetime.now().time())
    end = time.time()
    print("levou: "+str(end-start))

# Run the server's main loop
print("Aguardando requests")
server.serve_forever()

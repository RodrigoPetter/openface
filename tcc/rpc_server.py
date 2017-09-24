from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler
import datetime
import time
import subprocess
import infer

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

# Register a function under a different name
def exemplo(x,y):
    return x + y
server.register_function(exemplo)

def treinar():

    #raw_input("Treinamento 1")
    amostra = "final"
    gerarClassifier(amostra)
    
    with open("../treinamento_tcc/final/feature/classifier.pkl", "rb") as f:
        class1 = f.read().encode('base64')      

    return '{ "result":[{"nome":"Classifier1", "arquivo":"'+class1+'"}] }'
server.register_function(treinar)

def descobrir(treinamento64, imagem64):
    print("Imagem recebida. Iniciando escrita...")
    with open("../temp/imagem_temporaria.png", "wb") as f:
	    f.write(imagem64.decode('base64'))
    print("Escrita concluida.")
    
    print("Classificador recebido. Iniciando escrita...")
    with open("../temp/classifier.pkl", "wb") as f:
        f.write(treinamento64.decode('base64'))
    print("Escrita concluida.")
    
    result = infer.infer("../temp/classifier.pkl", "../temp/imagem_temporaria.png", True)
    print("Imagem Processada"+ result)
    return result
server.register_function(descobrir)

def gerarClassifier(amostra):
    workdir = "/root/openface/"
    
    print("Iniciando treinamento: "+amostra+" Foto")
    start = time.time()
    #raw_input("alinhamento")
    print(datetime.datetime.now().time())
    bashCommand = "./util/align-dlib.py ./treinamento_tcc/"+amostra+"/originais/ align outerEyesAndNose ./treinamento_tcc/"+amostra+"/alinhadas/ --size 96"
    p = subprocess.Popen(bashCommand, shell=True, cwd=workdir)
    p.wait()
    print(datetime.datetime.now().time())
    end = time.time()
    print("levou: "+str(end-start))

    #raw_input("representacao")
    print(datetime.datetime.now().time())         
    start = time.time()
    bashCommand = "./batch-represent/main.lua -outDir ./treinamento_tcc/"+amostra+"/feature/ -data ./treinamento_tcc/"+amostra+"/alinhadas/"
    p = subprocess.Popen(bashCommand, shell=True, cwd=workdir)
    p.wait()
    print(datetime.datetime.now().time())
    end = time.time()
    print("levou: "+str(end-start))

    #raw_input("classificador")
    print(datetime.datetime.now().time())
    start = time.time()
    bashCommand = "./demos/classifier.py train ./treinamento_tcc/"+amostra+"/feature/"
    p = subprocess.Popen(bashCommand, shell=True, cwd=workdir)
    p.wait()
    print(datetime.datetime.now().time())
    end = time.time()
    print("levou: "+str(end-start))

# Run the server's main loop
print("Aguardando requests")
server.serve_forever()

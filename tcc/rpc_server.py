from SimpleXMLRPCServer import SimpleXMLRPCServer
from SimpleXMLRPCServer import SimpleXMLRPCRequestHandler
import infer

# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

#Config
porta = 8081
endereco = "192.168.0.16"

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
    return "Nao implementado"
server.register_function(treinar)

def descobrir(base64):
    print("Imagem recebida. Iniciando escrita...")
    with open("../temp/imagem_temporaria.png", "wb") as f:
	f.write(base64.decode('base64'))
	print("Escrita concluida.")
    result = infer.infer("../models/openface/celeb-classifier.nn4.small2.v1.pkl", "../temp/imagem_temporaria.png", True)
    print("Imagem Processada"+ result)
    return result
server.register_function(descobrir)


# Run the server's main loop
print("Aguardando requests")
server.serve_forever()

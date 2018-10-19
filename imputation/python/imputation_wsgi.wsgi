import sys
import cgi
import cgitb
import pickle
cgitb.enable() 
python_path = '/usr/local/projects/imputation/python/'
project_path = '/usr/local/projects/imputation/project/'
sys.path.append(python_path)
import imputation_web 

                                                                                  
def application(environ, start_response):
    arguments = cgi.FieldStorage(environ=environ, fp=environ['wsgi.input'])
    dict_arg = {}
    for key in arguments.keys():        
       dict_arg[ key ] = arguments[ key ].value
       dict_arg[ key + '_filename' ] = arguments[ key ].filename           
    dict_arg['client_ip'] = get_client_address(environ)   
    pickle_out = open(project_path + "output/" + str(dict_arg.get('sessionid','none')) + "_" +  str(dict_arg['queryflag'])  +".pickle", "wb")
    pickle.dump(dict_arg, pickle_out) 
    pickle_out.close()         
    status = '200 OK'   
    JSON_Return = imputation_web.run_imputation(1,dict_arg)   
    #json = open(project_path + 'output/' + str(dict_arg.get('sessionid','none')) + "_" + str(dict_arg['queryflag']) + "_return.txt", 'w')
    #json.write(JSON_Return) 
    #json.close()            
    output = JSON_Return.encode()
    response_headers = [('Content-type', 'text/plain'),
                        ('Content-Length', str(len(output)))]
    start_response(status, response_headers) 
    return [output]  

    
def get_client_address(environ):
    try:
        return environ['HTTP_X_FORWARDED_FOR'].split(',')[-1].strip()
    except KeyError:
        return environ['REMOTE_ADDR']
from flask import Flask,render_template,request
import pickle
import numpy as np
# import sklearn
app=Flask(__name__)
model=pickle.load(open('pipe.pkl','rb'))
data=[]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=request.form.to_dict()
#     {'ram': '2', 'ssd': '0', 'hdd': '32', 'resolution': '1366x768', 'touchscreen': '1', 'IPS': '1',
#      'typename': 'Netbook', 'gpu': 'Intel', 'company': 'Samsung', 'cpu_company': 'Intel', 'opsys': 'Windows',
#      'inches': '13'}
    resx,resy = data['resolution'].split('x')
    inch=int(data['inches'])
    # x=[['Lenovo','Gaming',8,'Windows',0,1,157.98,'Intel Core i5',256,1000,'Nvidia']]

    brand=data['company']
    ram=int(data['ram'])
    ssd=int(data['ssd'])
    hdd=int(data['hdd'])
    gpu=data['gpu']
    cpu=data['cpu_company']
    opsys=data['opsys']
    touch=int(data['touchscreen'])
    ips=int(data['IPS'])
    type=data['typename']
    ppi=((( int(resx)**2 + int(resy)**2 ))**0.5/ inch)

    x=np.array([brand,type,ram,opsys,touch,ips,ppi,cpu,ssd,hdd,gpu])
    data=x
    print(x)
    x=x.reshape(1,11)
    prediction=str(int(np.exp(model.predict(x)[0])))
    return render_template('index.html',company=brand,prediction=prediction
                           ,ram=ram,ssd=ssd,hdd=hdd,gpu=gpu,
                           opsys=opsys,touchscreen=touch,IPS=ips,cpu_company=cpu,typename=type
                           )
if __name__ == '__main__':
    app.debug = True
    app.run()

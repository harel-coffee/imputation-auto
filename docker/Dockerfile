FROM ubuntu
RUN apt-get update
RUN apt-get install -y apt-utils build-essential vim curl git apache2 apache2-utils apache2-dev libapache2-mod-wsgi-py3 python3 python3-pip
RUN git clone --recursive https://github.com/dmlc/xgboost
WORKDIR /xgboost 
RUN make -j4
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install seaborn
RUN pip3 install sklearn
RUN pip3 install xgboost
RUN pip3 install gensim
RUN pip3 install pydotplus
RUN pip3 install tensorflow
RUN pip3 install keras
RUN pip3 install lxml 
COPY ./imputation_vhost.conf /etc/apache2/sites-available/imputation_vhost.conf
RUN a2ensite imputation_vhost.conf
WORKDIR /
CMD ["apache2ctl", "-D", "FOREGROUND"]





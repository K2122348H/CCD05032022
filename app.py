#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask


# In[2]:


app = Flask(__name__)


# In[3]:


from flask import render_template, request 
import joblib

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        income = request.form.get("income")
        age = request.form.get("age")
        loan = request.form.get("loan")
        income = float(income)
        age = float(age)
        loan = float(loan)
        print(income, age, loan)
        
        model1 = joblib.load("CCD_REG")
        pred1 = model1.predict([[income, age, loan]])
        s1 = "The score of credit card default based on Linear Regression is " + str(pred1[0])
        
        model2 = joblib.load("CCD_DT")
        pred2 = model2.predict([[income, age, loan]])
        s2 = "The score of credit card default based on Decision Tree is " + str(pred2[0])
        
        model3 = joblib.load("CCD_NN")
        pred3 = model3.predict([[income, age, loan]])
        s3 = "The score of credit card default based on Neural Network is " + str(pred3[0])
        
        model4 = joblib.load("CCD_GB")
        pred4 = model4.predict([[income, age, loan]])
        s4 = "The score of credit card default based on Gradient Boosting is " + str(pred4[0])
        
        model5 = joblib.load("CCD_RF")
        pred5 = model5.predict([[income, age, loan]])
        s5 = "The score of credit card default based on Random Forest is " + str(pred5[0])
        
        return(render_template("index.html", result1=s1,result2=s2,result3=s3,result4=s4,result5=s5))
    else:
        return(render_template("index.html", result1="No input",result2="No input",result3="No input",result4="No input",result5="No input"))


# In[ ]:


if __name__=="__main__":
    app.run()


# In[ ]:





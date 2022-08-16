from flask import Flask, request, render_template
import pandas as pd
'''import warnings
from decimal import Decimal
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
print('1')
from sklearn.model_selection import cross_val_score
from statistics import mean
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from itertools import combinations
from collections import Counter
import operator
from sklearn.linear_model import LogisticRegression
import json'''

print('2')
'''warnings.simplefilter("ignore")
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')'''
app = Flask(__name__)
'''df_comb = pd.read_csv("dataset/dis_sym_dataset_dr_comb.csv") # Disease combination
X_train = df_comb.iloc[:, 1:]
Y_train = df_comb.iloc[:, 0:1]'''

'''rf =   LogisticRegression()
rf = rf.fit(X_train, Y_train)
scores = cross_val_score(rf, X_train, Y_train, cv=3)
score = round(Decimal(scores.mean()*100),2)


test=pd.read_csv("dataset/dis_sym_dataset_dr_norm.csv",error_bad_lines=False)
X_test = test.iloc[:, 1:]
Y_test = test.iloc[:, 0:1]
x_test=test.drop('label_dis',axis=1)

# List of symptoms
dataset_symptoms = list(X_test.columns)'''

print('3')
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/suggest', methods=['POST', 'GET'])
def suggest():
    print('------>', 'suggest function called')
    if request.method == 'POST':
        name = request.args.get('name')
        print('post data name ---->', name)
        return 'API POST data...'
    return 'This is my first API call!'

# #if gender is male of any age and they have entered a symptom which also present in female disease than system should not recommend female disease
# # eg: Symptom is blurred vision and it should give output as Diabetes Mellitus but not Eclampsia and Pre-eclampsia
@app.route('/male_suggest', methods=['POST', 'GET'])
def male_suggest():
    temp = 'male suggest data...'
    return temp

# #cosider this api for symtom suggestion for female as gender of any age
# #eg: Symptom is blurred vision and it should give output as Diabetes Mellitus, Eclampsia and Pre-eclampsia
# @app.route('/female_suggest', methods=['POST', 'GET'])
# def female_suggest():
#     if request.method == 'POST':
#         col = x_test.columns
#         inputt = [str(x) for x in request.form.values()]
#         processed_user_symptoms = []
#         for sym in inputt:
#             sym = sym.strip()
#             sym = sym.replace('-', ' ')
#             sym = sym.replace("'", '')
#             sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
#             processed_user_symptoms.append(sym)
#         user_symptoms = []
#         for user_sym in inputt:
#             user_sym = user_sym.split()
#             str_sym = set()
#             for comb in range(1, len(user_sym) + 1):
#                 for subset in combinations(user_sym, comb):
#                     subset = ' '.join(subset)
#                     str_sym.update(subset)
#             str_sym.add(' '.join(user_sym))
#             user_symptoms.append(' '.join(user_sym).replace('_', ' '))
#         found_symptoms = set()
#         for idx, data_sym in enumerate(dataset_symptoms):
#             data_sym_split = data_sym.split()
#             for user_sym in user_symptoms:
#                 count = 0
#                 for symp in data_sym_split:
#                     if symp in user_sym.split():
#                         count += 1
#                 if count / len(data_sym_split) > 0.5:
#                     found_symptoms.add(data_sym)
#         found_symptoms = list(found_symptoms)
#         dis_list = set()
#         final_symp = []
#         counter_list = []
#         for idx, symp in enumerate(found_symptoms):
#             symptom = found_symptoms[int(idx)]
#             final_symp.append(symptom)
#             dis_list.update(set(test[test[symptom] == 1]['label_dis']))
#         print(dis_list)
#         for dis in dis_list:
#             row = test.loc[test['label_dis'] == dis].values.tolist()
#             row[0].pop(0)
#             for idx, val in enumerate(row[0]):
#                 if val != 0 and dataset_symptoms[idx] not in final_symp:
#                     counter_list.append(dataset_symptoms[idx])
#         dict_symp = dict(Counter(counter_list))
#         dict_symp_tup = sorted(dict_symp.items(), key=operator.itemgetter(1), reverse=True)
#         found_suggested_symptoms = []
#         for tup in dict_symp_tup:
#             if 'female' not in tup:
#                 found_suggested_symptoms.append(tup[0])
#         found_symptoms_10 = found_suggested_symptoms[0:11]
#         data_suggest = {}
#         data_suggest['female_suggested_symtom'] = found_symptoms_10
#         print(data_suggest)
#         return data_suggest

# #api for age as (adult/child/infant/elderly) and with gender as female
# #developer need to pass age with list of symptom passed to it.
# #eg: If someone passed, [child, headache] as a input, than it should give disease related to headache but not disease like
# # high blood pressure because its a case of adult
# @app.route('/female_age_suggest', methods=['POST', 'GET'])
# def female_age_suggest():
#     if request.method == 'POST':
#         col = x_test.columns
#         inputt = [str(x) for x in request.form.values()]
#         processed_user_symptoms = []
#         for sym in inputt:
#             sym = sym.strip()
#             sym = sym.replace('-', ' ')
#             sym = sym.replace("'", '')
#             sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
#             processed_user_symptoms.append(sym)
#         user_symptoms = []
#         for user_sym in inputt:
#             user_sym = user_sym.split()
#             str_sym = set()
#             for comb in range(1, len(user_sym) + 1):
#                 for subset in combinations(user_sym, comb):
#                     subset = ' '.join(subset)
#                     str_sym.update(subset)
#             str_sym.add(' '.join(user_sym))
#             user_symptoms.append(' '.join(user_sym).replace('_', ' '))
#         found_symptoms = set()
#         for idx, data_sym in enumerate(dataset_symptoms):
#             data_sym_split = data_sym.split()
#             for user_sym in user_symptoms:
#                 count = 0
#                 for symp in data_sym_split:
#                     if symp in user_sym.split():
#                         count += 1
#                 if count / len(data_sym_split) > 0.5:
#                     found_symptoms.add(data_sym)
#         found_symptoms = list(found_symptoms)
#         dis_list = set()
#         final_symp = []
#         counter_list = []
#         age_list = ['child', 'adult', 'infant', 'elderly']
#         final_age_list = []
#         input_age_list = []
#         for idx, symp in enumerate(found_symptoms):
#             symptom = found_symptoms[int(idx)]
#             print(symptom)
#             if symptom not in age_list:
#                 final_symp.append(symptom)
#                 dis_list.update(set(test[test[symptom] == 1]['label_dis']))
#             else:
#                 input_age_list.append(symptom)
#         for age in age_list:
#             if age not in input_age_list:
#                 final_age_list.append(age)
#         # print("Input Age List: ",input_age_list)
#         # print("Dis List: ",dis_list)
#         # print("Final Symp: ",final_symp)
#         # print("Final Age List: ",final_age_list)
#         # # print(dis_list)  #whole disease with symptom and entered age
#         # #print(age_dis_list)  #whole disease excluding entered age
#         # #i want whole symptom disease list excluding non entered age
#         # #print(tes) #agr headache and child add kiya h.. to suggested disease list m high blood pressure ni aana chahiye
#         ww = []
#         xx = []
#         v = []
#         for dis in dis_list:
#             row = test.loc[test['label_dis'] == dis].values.tolist()
#             row[0].pop(0)
#             for idx, val in enumerate(row[0]):
#                 if val != 0 and dataset_symptoms[idx] not in final_symp:
#                     if dataset_symptoms[idx] not in final_age_list:
#                         ww.append(dis)
#                         for present in ww:
#                             if present not in v:
#                                 v.append(present)
#                     else:
#                         # exclude this disease
#                         xx.append(dis)
#         for z in xx:
#             v.remove(z)
#         print("Disease List: ", v)
#         for dis in v:
#             row = test.loc[test['label_dis'] == dis].values.tolist()
#             row[0].pop(0)
#             for idx, val in enumerate(row[0]):
#                 if val != 0 and dataset_symptoms[idx] not in final_symp:
#                     if dataset_symptoms[idx] not in age_list:
#                         counter_list.append(dataset_symptoms[idx])
#         dict_symp = dict(Counter(counter_list))
#         dict_symp_tup = sorted(dict_symp.items(), key=operator.itemgetter(1), reverse=True)
#         found_suggested_symptoms = []
#         for tup in dict_symp_tup:
#             found_suggested_symptoms.append(tup[0])
#         found_symptoms_10 = found_suggested_symptoms[0:11]
#         data_suggest = {}
#         data_suggest['female_age_suggested_symtom'] = found_symptoms_10
#         print(data_suggest)
#         return data_suggest

# #api for age as (adult/child/infant/elderly) and with gender as male
# #developer need to pass age with list of symptom passed to it.
# #eg: If someone passed, [elderly, depression, headache] as a input, than it should give disease related to headache and depression but not disease like
# # Pre-eclampsia because its a case of female and 'hypertension / high blood pressure because of adult as age
# @app.route('/male_age_suggest', methods=['POST', 'GET'])
# def male_age_suggest():
#     if request.method == 'POST':
#         col = x_test.columns
#         inputt = [str(x) for x in request.form.values()]
#         processed_user_symptoms = []
#         for sym in inputt:
#             sym = sym.strip()
#             sym = sym.replace('-', ' ')
#             sym = sym.replace("'", '')
#             sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
#             processed_user_symptoms.append(sym)
#         user_symptoms = []
#         for user_sym in inputt:
#             user_sym = user_sym.split()
#             str_sym = set()
#             for comb in range(1, len(user_sym) + 1):
#                 for subset in combinations(user_sym, comb):
#                     subset = ' '.join(subset)
#                     str_sym.update(subset)
#             str_sym.add(' '.join(user_sym))
#             user_symptoms.append(' '.join(user_sym).replace('_', ' '))
#         found_symptoms = set()
#         for idx, data_sym in enumerate(dataset_symptoms):
#             data_sym_split = data_sym.split()
#             for user_sym in user_symptoms:
#                 count = 0
#                 for symp in data_sym_split:
#                     if symp in user_sym.split():
#                         count += 1
#                 if count / len(data_sym_split) > 0.5:
#                     found_symptoms.add(data_sym)
#         found_symptoms = list(found_symptoms)
#         dis_list = set()
#         final_symp = []
#         counter_list = []
#         age_list = ['child', 'adult', 'infant', 'elderly']
#         final_age_list = []
#         input_age_list = []
#         female_dis_list = []
#         for idx, symp in enumerate(found_symptoms):
#             symptom = found_symptoms[int(idx)]
#             if symptom not in age_list:
#                 final_symp.append(symptom)
#             else:
#                 input_age_list.append(symptom)
#             dis_list.update(set(test[test[symptom] == 1]['label_dis']))
#         for age in age_list:
#             if age not in input_age_list:
#                 final_age_list.append(age)
#         ww = []
#         xx = []
#         v = []
#         for dis in dis_list:
#             row = test.loc[test['label_dis'] == dis].values.tolist()
#             row[0].pop(0)
#             for idx, val in enumerate(row[0]):
#                 if val != 0 and dataset_symptoms[idx] not in final_symp:
#                     if 'female' not in dataset_symptoms[idx]:
#                         if dataset_symptoms[idx] not in final_age_list:
#                             ww.append(dis)
#                             for present in ww:
#                                 if present not in v:
#                                     v.append(present)
#                         else:
#                             # exclude this disease because of age which is not cosidered in given age.
#                             xx.append(dis)
#                     else:
#                         female_dis_list.append(dis)

#         for age_dis in xx:
#             v.remove(age_dis)
#         for female_dis in female_dis_list:
#             v.remove(female_dis)
#         print("Final Disease List: ", v)
#         # print("    ")
#         # print("Disease which occur to female: ", female_dis_list)
#         # print("   ")
#         # print("Disease which occur to age group other than inputted age: ",xx)
#         for dis in v:
#             row = test.loc[test['label_dis'] == dis].values.tolist()
#             row[0].pop(0)
#             for idx, val in enumerate(row[0]):
#                 if val != 0 and dataset_symptoms[idx] not in final_symp:
#                     if dataset_symptoms[idx] not in age_list:
#                         counter_list.append(dataset_symptoms[idx])
#         dict_symp = dict(Counter(counter_list))
#         dict_symp_tup = sorted(dict_symp.items(), key=operator.itemgetter(1), reverse=True)
#         found_suggested_symptoms = []
#         for tup in dict_symp_tup:
#             found_suggested_symptoms.append(tup[0])
#         found_symptoms_10 = found_suggested_symptoms[0:11]
#         data_suggest = {}
#         data_suggest['male_age_suggested_symtom'] = found_symptoms_10
#         print(data_suggest)
#         return data_suggest

# @app.route('/predict', methods=['POST', 'GET'])
# def predict():
#     if request.method == 'POST':
#         col = x_test.columns
#         inputt = [str(x) for x in request.form.values()]
#         processed_user_symptoms = []
#         for sym in inputt:
#             sym = sym.strip()
#             sym = sym.replace('-', ' ')
#             sym = sym.replace("'", '')
#             sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
#             processed_user_symptoms.append(sym)
#         user_symptoms = []
#         for user_sym in inputt:
#             user_sym = user_sym.split()
#             str_sym = set()
#             for comb in range(1, len(user_sym) + 1):
#                 for subset in combinations(user_sym, comb):
#                     subset = ' '.join(subset)
#                     str_sym.update(subset)
#             str_sym.add(' '.join(user_sym))
#             user_symptoms.append(' '.join(user_sym).replace('_', ' '))
#         found_symptoms = set()
#         for idx, data_sym in enumerate(dataset_symptoms):
#             data_sym_split = data_sym.split()
#             for user_sym in user_symptoms:
#                 count = 0
#                 for symp in data_sym_split:
#                     if symp in user_sym.split():
#                         count += 1
#                 if count / len(data_sym_split) > 0.8:
#                     found_symptoms.add(data_sym)
#         found_symptoms = list(found_symptoms)
#         sample_x = [0 for x in range(0, len(dataset_symptoms))]
#         for val in found_symptoms:
#             sample_x[dataset_symptoms.index(val)] = 1
#         rf_1 =  LogisticRegression()
#         rf_1 = rf_1.fit(X_test, Y_test)
#         score_1 = rf_1.score(X_test, Y_test)
#         prediction = rf_1.predict_proba([sample_x])
#         predictions = round(Decimal(prediction.mean() * 100), 2)
#         k = 1#3
#         diseases = list(set(Y_test['label_dis']))
#         diseases.sort()
#         topk = prediction[0].argsort()[-k:][::-1]
#         topk_dict = {}
#         diseaseList = list()
#         for idx, t in enumerate(topk):
#             diseaseList.append(diseases[t])
#             match_sym = set()
#             row = test.loc[test['label_dis'] == diseases[t]].values.tolist()
#             row[0].pop(0)
#             for idx, val in enumerate(row[0]):
#                 if val != 0:
#                     match_sym.add(dataset_symptoms[idx])
#             prob = (len(match_sym.intersection(set(found_symptoms))) + 1) / (len(set(found_symptoms)) + 1)
#             prob *= mean(scores)
#             topk_dict[t] = prob
#         prob11 = list(topk_dict.values())[0]
#         prob22 = list(topk_dict.values())[1]
#         prob33 = list(topk_dict.values())[2]
#         data = {}
#         data['disease_list'] = diseaseList[0]#, diseaseList[1], diseaseList[2]
#         data['prob'] = prob11 * 100#, prob22 * 100, prob33 * 100
#         return (data)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)

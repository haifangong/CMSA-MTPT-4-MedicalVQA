import json

data_path = 'F:/Medical-VQA/MICCAI19-MedVQA/data_RAD/testset.json'
data = json.load(open(data_path))

all_ques = []
abdomen = []
brain = []
chest = []

for i in range(len(data)):
    x = data[i]
    ques = x['question']
    modal = x['image_organ']
    if ques not in all_ques:
        all_ques.append(ques)
    if modal == 'ABD':
        if ques not in abdomen:
            abdomen.append(ques)
    if modal == 'HEAD':
        if ques not in brain:
            brain.append(ques)
    if modal == 'CHEST':
        if ques not in chest:
            chest.append(ques)

abd_lst = []
brain_lst = []
chest_lst = []
for i in range(len(all_ques)):
    ques = all_ques[i]
    abd_lbl, brain_lbl, chest_lbl = 0, 0, 0
    if ques in abdomen:
        abd_lbl = 1
    if ques in brain:
        brain_lbl = 1
    if ques in chest:
        chest_lbl = 1
    abd_item = {'question': ques, 'label': abd_lbl}
    brain_item = {'question': ques, 'label': brain_lbl}
    chest_item = {'question': ques, 'label': chest_lbl}
    abd_lst.append(abd_item)
    brain_lst.append(brain_item)
    chest_lst.append(chest_item)

with open("abdomen_test.json","w") as f:
    json.dump(abd_lst,f)
with open("brain_test.json","w") as f:
    json.dump(brain_lst,f)
with open("chest_test.json","w") as f:
    json.dump(chest_lst,f)

# data_path = './abdomen_train.json'
# data = json.load(open(data_path))
# print(data[0]['label'])
# print(type(data[0]['label']))

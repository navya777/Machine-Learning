import sys
import random
import math


############dot product############

def dotproduct(u, v):
    assert len(u) == len(v), "dotproduct: u and v must be of same length"
    dp = 0
    for i in range(0, len(u), 1):
        dp += u[i] * v[i]
    return dp


#####standardize the feature space and Orginal data##########

def standardize_data(train_set, test_set):
    wlen = []
    ecl = []
    rt = len(train_set)
    rtest = len(test_set)
    col_std = len(train_set[0])
    col_tes = len(test_set[0])
    #print("coloum values",no_hyperplane,col_std,col_tes)
    for j in range(0, col_std, 1):
        wlen.append(0)
        ecl.append(0)
    for j in range(0, col_std, 1):
        for i in range(0, rt, 1):
            wlen[j] += (train_set[i][j] ** 2)
        ecl[j] = math.sqrt(wlen[j])


    ####to account for divison by zero
    for j in range(0, col_std, 1):
        if ecl[j] == 0:
            ecl[j] = 1


    for t in range(0,rt,1):
        for h in range(0,col_std,1):
            train_set[t][h] = train_set[t][h] / ecl[h]
        train_set[t].append(1)



    for q in range(0, rtest, 1):
        for w in range(0,col_tes, 1):
            test_set[q][w] = test_set[q][w] / ecl[w]
        test_set[q].append(1)

    return train_set, test_set


#####sign function

def sign(a):
    if a > 0:
        return 1
    else:
        return -1


####################hinge loss###################

def hinge_loss(train_data, train_label):
    col = len(train_data[0])
    ro = len(train_data)
    eta = .001
    prevobj = 1000000
    obj = 10
    last = 0
    w4 = []
    for j in range(0, col, 1):
        w4.append(random.uniform(-0.1, 0.1))
    while abs(prevobj - obj) > .01:
        # Compute differential (dell_f)
        prevobj = obj
        dell_f = []
        for j in range(0, col, 1):
            dell_f.append(0)

        for i in range(0, ro, 1):
            dp = 0
            dp = dotproduct(w4, train_data[i])
            gradient = train_label[i] * dp
            if gradient < 1:
                for j in range(0,col, 1):
                    dell_f[j] = dell_f[j] + (train_label[i]) * float(train_data[i][j])

        # Update w
        for j in range(0, col, 1):
            w4[j] = (w4[j]) + eta * (dell_f[j])
            last = j
        w0 = w4[last]

        # Compute Error
        error = 0.0
        for i in range(0, ro, 1):
            hingeloss = 1 - float(train_label[i] * dotproduct(w4, train_data[i]))
            if (hingeloss < 0):
                error = error + 0
            else:
                error += hingeloss

        obj = error
    return w4, w0


###################
#### Classify unlabeled points
##################

def prediction_hinge(w,test_prediction):
    pred_result = []
    for i in range(0, row_test_hinge, 1):
        dotp = dotproduct(w, test_prediction[i])
        if dotp > 0:
            # print("1", i)
            pred_result.append(1)
        else:
            # print("-1", i)
            pred_result.append(-1)
    return pred_result


########Read data using training dataset


file = sys.argv[1]
datafile = open(file)
data = []
l = datafile.readline()
train_label = []

while l != '':
    val = l.split()
    l1 = []
    for j in range(1, len(val), 1):
        l1.append(float(val[j]))
    data.append(l1)
    train_label.append(int(val[0]))
    l = datafile.readline()

rows = len(data)
colomn = len(data[0])
datafile.close()
#print(colomn)
#########read test data
testfile = sys.argv[2]
test_file = open(testfile)
testdata = []
t = test_file.readline()

while t != '':
    tes = t.split()
    l2 = []
    for j in range(1, len(tes), 1):
        l2.append(float(tes[j]))
    testdata.append(l2)
    t = test_file.readline()

test_row = len(testdata)
test_col = len(testdata[0])
#print(test_col)
no_hyperplane = 0
no_hyperplane = int(sys.argv[3])     ###hyper plane input

feature_space_tran = []
feature_space_test = []
for i in range(0, no_hyperplane, 1):
    w = []
    for r in range(0, colomn, 1):
        w.append(random.uniform(-1, 1))    ###random weight between -1 and 1
    #print('value w', len(w), len(data[0]))

    sum_Weight = []
    for u in range(0, rows, 1):
        sum_Weight.append(0)
    for p in range(0, rows, 1):
        sum_Weight[p] = dotproduct(data[p], w)
    min_w = min(sum_Weight)
    max_w = max(sum_Weight)                 ####min and max weight to find out w0

    z = []
    for s in range(0, rows, 1):
        z.append(0)
    zt = []
    for u in range(0, test_row, 1):
        zt.append(0)
    w0 = random.uniform(min_w, max_w)
    for j in range(0, rows, 1):
        z[j] = dotproduct(data[j], w) + w0
        z[j] = int((1 + sign(z[j])) / 2)        #######creating new feature space taking sign activation function
    feature_space_tran.append(z)
    for l in range(0, test_row, 1):
        zt[l] = dotproduct(testdata[l], w) + w0
        zt[l] = int((1 + sign(zt[l])) / 2)      #######creating new feature space taking sign activation function
    feature_space_test.append(zt)

feature_row = no_hyperplane
feature_Col_test = len(feature_space_test[0])
feature_col_tran = len(feature_space_tran[0])
#print('row ,col', feature_row, len(feature_space_tran[0]), feature_Col_test, feature_col_tran)
z_traindata = []
zt_testdata = []
z_traindata = [list(a) for a in (zip(*feature_space_tran))]   ####new feature space where coloumn is no of hyperplane
zt_testdata = [list(b) for b in zip(*feature_space_test)]       #####and row is orginal row number in train/test data
train_data = []
test_data = []
[train_data, test_data] = standardize_data(z_traindata, zt_testdata)   ###standardising the feature space for faster execution
ro_hinge = len(train_data)
col_hinge = len(train_data[0])
row_test_hinge = len(test_data)
[wt_hi, w0_hi] = hinge_loss(train_data, train_label)  ####applying hinge on feature space after std
hinge_Pre = prediction_hinge(wt_hi,test_data)
OUT = open("01space_output.txt", 'w')       #####prediction on feature space
OUT.write(str(hinge_Pre))
OUT.close()
[train_data_hinge,test_data_hinge] = standardize_data(data,testdata)  ###std of orginal data for hinge loss
[w_hinge,w0_hinge] = hinge_loss(train_data_hinge,train_label)       ####executing hinge on std orginal data
hinge_data = prediction_hinge(w_hinge,test_data_hinge)          ####hinge loss predction on orginal data
OUT = open("original_output.txt",'w')
OUT.write(str(hinge_data))
OUT.close()


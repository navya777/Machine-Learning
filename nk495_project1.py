import sys
import random
import math
import copy

#### FUNCTIONS #########

###################
#### Supplementary function to compute dot product
###################
def dotproduct(u, v):
    assert len(u) == len(v), "dotproduct: u and v must be of same length"
    dp = 0
    for i in range(0, len(u), 1):
        dp += u[i] * v[i]
    return dp


###################
## Standardize the code here: divide each feature of each
## datapoint by the length of each column in the training data
## return [traindata, testdata]
###################
def standardize_data(train_data, test_data):
    wlen = []
    ecl = []

    for j in range(0, cols, 1):
        wlen.append(0)
        ecl.append(0)
    for j in range(0, cols-1, 1):
        for i in range(0, rows, 1):
            wlen[j] += (train_data[i][j] ** 2)
        ecl[j] = math.sqrt(wlen[j])
        # print(ecl[j])

    ####to account for divison by zero
    for j in range(0, cols, 1):
        if ecl[j] == 0:
            ecl[j] = 1

    for i in range(0, rows, 1):
        for j in range(0, cols-1, 1):
            train_data[i][j] = train_data[i][j] / ecl[j]
        train_data[i].append(1)
    for i in range(0, test_row, 1):
        for j in range(0, test_col-1, 1):
            test_data[i][j] = test_data[i][j] / ecl[j]
        test_data[i].append(1)
    # cols = len(traindata[0])
    # test_col = len(traindata[0])
    rows_std=len(train_data)
    cols_std=len(train_data[0])
    print("after std",rows_std,cols_std)
    print("aff",traindata[0])
    print("fff",testdata[0])
    return train_data, test_data


###################
## Solver for least squares (linear regression):
## return [w, w0]
###################
def least_squares(traindata, trainlabel):
    eta = 0.001
    prev_obj = 10000
    obj = 10
    origin = 0
    w2 = []
    for j in range(0, cols_std, 1):
        w2.append(random.uniform(-0.1, 0.1))

    while abs(prev_obj - obj) > 0.01:
        prev_obj = obj
        diff = []
        for j in range(0, cols_std, 1):
            diff.append(0)

        for i in range(0, rows_std, 1):
            dotp = 0
            dotp = dotproduct(w2, traindata[i])
            for j in range(0, cols_std, 1):
                diff[j] = diff[j] + (trainlabel[i] - dotp) * float(traindata[i][j])

        for j in range(0, cols_std, 1):
            w2[j] = (w2[j]) + eta * (diff[j])
            origin = j
        w0 = w2[origin]
        error = 0
        for i in range(0, rows_std, 1):
            dot = dotproduct(w2, traindata[i])
            error += (trainlabel[i] - dot) ** 2
        obj = error

    return w2, w0


###################
## Solver for regularized least squares (linear regression)
## return [w, w0]
###################
def least_squares_regularized(traindata, trainlabel):
    eta = 0.001
    prev_obj = 10000
    obj = 10

    lam = 0.01
    last = 0
    w3 = []
    w_len1 = 0

    for j in range(0, cols_std, 1):
        w3.append(random.uniform(-0.1, 0.1))
        w_len1 += w3[j] ** 2

    abs_w1 = math.sqrt(w_len1)
    while abs(prev_obj - obj) > 0.01:
        prev_obj = obj
        diff = []
        for j in range(0, cols_std, 1):
            diff.append(0)

        for i in range(0, rows_std, 1):
            dotp = 0
            dotp = dotproduct(w3, traindata[i])
            for j in range(0, cols_std, 1):
                diff[j] = diff[j] + ((trainlabel[i] - dotp) * float(traindata[i][j]) + 2 * lam * abs_w1)

        for j in range(0, cols_std, 1):
            w3[j] = (w3[j]) + eta * (diff[j])
            last = j

        least_regerror = 0
        for i in range(0, rows_std, 1):
            dot = dotproduct(w3, traindata[i])
            least_regerror += ((trainlabel[i] - dot) ** 2 + lam * math.pow(abs_w1, 2))
        obj = least_regerror

    w0 = w3[last]
    return w3, w0


###################
## Solver for hinge loss
## return [w, w0]
###################
def hinge_loss(traindata, trainlabel):
    eta = .001
    prevobj = 1000000
    obj = 10
    last = 0
    w4 = []
    for j in range(0, cols_std, 1):
        w4.append(random.uniform(-0.1, 0.1))
    while abs(prevobj - obj) > .01:
        # Compute differential (dell_f)
        prevobj = obj
        dell_f = []
        for j in range(0, cols_std, 1):
            dell_f.append(0)

        for i in range(0, rows_std, 1):
            dp = 0
            dp = dotproduct(w4, traindata[i])
            gradient = trainlabel[i] * dp
            if gradient < 1:
                for j in range(0, cols_std, 1):
                    dell_f[j] = dell_f[j] + (trainlabel[i]) * float(traindata[i][j])

        # Update w
        for j in range(0, cols_std, 1):
            w4[j] = (w4[j]) + eta * (dell_f[j])
            last = j
        w0 = w4[last]

        # Compute Error
        error = 0.0
        for i in range(0, rows_std, 1):
            hingeloss = 1 - float(trainlabel[i] * dotproduct(w4, traindata[i]))
            if (hingeloss < 0):
                error = error + 0
            else:
                error += hingeloss

        obj = error
    return w4, w0


###################
## Solver for regularized hinge loss
## return [w, w0]
###################
def hinge_loss_regularized(traindata, trainlabel):
    eta = .001
    prevobj = 1000000
    obj = 10
    lam = 0.01
    norm_w = 0
    last = 0
    w5 = []

    for j in range(0, cols_std, 1):
        w5.append(random.uniform(-0.1, 0.1))
    while abs(prevobj - obj) > .01:
        # Compute differential (dell_f)
        dell_f = []
        for j in range(0, cols_std, 1):
            dell_f.append(0)

        for j in range(0, cols_std - 1, 1):
            norm_w = norm_w + w5[j] ** 2

        norm_w = math.sqrt(norm_w)
        for i in range(0, rows_std, 1):
            dp = 0
            dp = dotproduct(w5, traindata[i])
            gradient = trainlabel[i] * dp
            if gradient < 1:
                for j in range(0, cols_std, 1):
                    dell_f[j] = dell_f[j] + (trainlabel[i]) * float(traindata[i][j]) + 2 * lam * w5[j]
            else:
                for j in range(0, cols_std, 1):
                    dell_f[j] += 2 * lam * w5[j]

        # Update w
        for j in range(0, cols_std, 1):
            w5[j] = (w5[j]) + eta * (dell_f[j])
            last = j
        w0 = w5[last]

        # Compute Error
        error = 0.0
        for i in range(0, rows_std, 1):
            hingeloss = 1 - float(trainlabel[i] * dotproduct(w5, traindata[i]))
            if (hingeloss < 0):
                error += lam * (norm_w ** 2)
            else:
                error += hingeloss + lam * (norm_w ** 2)
        prevobj = obj
        obj = error
    return w5, w0


###################
## Solver for logistic regression
## return [w, w0]
###################
def logistic_loss(traindata, trainlabel):
    wlen = 0
    w1 = []
    for j in range(0, cols_std, 1):
        w1.append(random.uniform(-0.1, 0.1))
        wlen += w1[j] ** 2
    #print("stadn",traindata[0])
    for i in range(0,rows_std, 1):
        if trainlabel[i] == -1:
            trainlabel[i] = 0
    eta = .001
    prevobj = 1000000
    obj = 10
    lam = 0.01
    normw = 0
    last = 0
    while abs(prevobj - obj) > .01:
        # Compute differential (dell_f)
        prevobj = obj
        dell_f = []
        for j in range(0, cols_std, 1):
            dell_f.append(0)
        for i in range(0, rows_std, 1):
            dp = 0.0
            dp = dotproduct(w1, traindata[i])

            for j in range(0, cols_std, 1):
                dell_f[j] += (trainlabel[i] - (1.0 / (1 + math.exp(-dp)))) * traindata[i][j]
        for j in range(cols_std):
            w1[j] += eta * dell_f[j]
            last = j

        #	print(w)
        error = 0.0
        for i in range(rows_std):
            y = dotproduct(w1, traindata[i])
            if trainlabel[i] == 0:
                error += -1 * math.log(1 - (1.0 / (1 + math.exp(-y))))
            else:
                error += -1 * math.log(1.0 / (1 + math.exp(-y)))

        obj = error
    w0 = w1[last]
    return w1, w0


###################
## Solver for adaptive learning rate hinge loss
## return [w, w0]
###################
def hinge_loss_adaptive_learningrate(traindata, trainlabel):
    prevobj = 1000000
    obj = 10
    w6 = []
    for i in range(0,rows,1):
        traindata[i].append(1)
    for j in range(0, len(traindata[0]), 1):
        w6.append(random.uniform(-0.1, 0.1))
    print("length W",len(w6))
    print("length train", len(traindata))
    print("over here",traindata[0])
    while abs(prevobj - obj) > .001:
        # Compute differential (dell_f)
        prevobj = obj
        dell_f = []
        last = 0
        for j in range(0, cols, 1):
            dell_f.append(0)
            last=j

        for i in range(0, rows, 1):
            a = trainlabel[i] * dotproduct(w6, traindata[i])
            for j in range(0, cols,1):
                if a < 1:
                    dell_f[j] += -(trainlabel[i] * traindata[i][j])
                else:
                    dell_f[j] += 0

        eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001,
                    .00000000001]
        eta_error = 1000000000000

        for r in range(0, len(eta_list), 1):
            eta = eta_list[r]

            # update w
            for j in range(0, cols, 1):
                w6[j] -= eta * dell_f[j]
            ###calculate error
            error = 0.0
            for i in range(0, rows, 1):
                hingeloss = 1 - float(trainlabel[i] * dotproduct(w6, traindata[i]))
                if (hingeloss < 0):
                    error = error + 0
                else:
                    error += hingeloss

            obj = error
            if obj < eta_error:
                eta_error = obj
                best_eta = eta
                # update w
                for j in range(0, cols, 1):
                    w[j] += eta * dell_f[j]

        if best_eta != None:
            eta = best_eta

            # update w
            for j in range(0, cols, 1):
                w6[j] -= eta * dell_f[j]
        error = 0.0
        for i in range(0, rows, 1):
            hingeloss = 1 - float(trainlabel[i] * dotproduct(w6, traindata[i]))
            if (hingeloss < 0):
                error = error + 0
            else:
                error += hingeloss
        obj = error
        print('best eta', best_eta)
        w0=w6[last]
    return w6,w0


#### MAIN #########

###################
#### Code to read train data and train labels
###################
data = sys.argv[1]
testdataset = sys.argv[2]

# Opening files and reading training data and lable data
data_file = open(data)
traindata = []
trainlabel = []
l_file = data_file.readline()
while (l_file != ''):
    a = l_file.split()
    l1 = []
    for j in range(1, len(a), 1):
        l1.append(float(a[j]))
    trainlabel.append(int(a[0]))
    traindata.append(l1)
    l_file = data_file.readline()
rows = len(traindata)
cols = len(traindata[0])
data_file.close()


before_stdtrain = []
before_stdtrain=copy.deepcopy(traindata)
test_da = open(testdataset)
testdata = []

l_test = test_da.readline()

while l_test != '':
    b = l_test.split()
    l2 = []
    for j in range(1, len(b), 1):
        l2.append(float(b[j]))
    testdata.append(l2)
    l_test = test_da.readline()
test_row = len(testdata)
test_col = len(testdata[0])

rows_std=0
cols_std=0
###################
#### Code to test data and test labels
#### The test labels are to be used
#### only for evaluation and nowhere else.
#### When your project is being graded we
#### will use 0 label for all test points
###################
print(traindata[0])
[std_traindata, std_testdata] = standardize_data(traindata, testdata)

rows_std=len(std_traindata)
cols_std=len(std_traindata[0])
rows_teststd=len(std_testdata)
cols_teststd=len(std_testdata[0])
print(rows,cols,rows_std,cols_std)
w = []
# for j in range(0, cols, 1):
#     w.append(0)
wlen = 0
abs_w = 0.0
# print('length of w',len(w))
for j in range(0, cols, 1):
    w.append(random.uniform(-0.1, 0.1))
    wlen += w[j] ** 2
# print('length of w',len(w))
abs_w = math.sqrt(wlen)
[w_hinge_Adaptive,w0]= hinge_loss_adaptive_learningrate(before_stdtrain, trainlabel)
[w_l, w0] = least_squares(std_traindata, trainlabel)
# print('least square W',w_l)
[w_lr, w0] = least_squares_regularized(std_traindata, trainlabel)
# print("least_regularised",w_lr)
[w_hi, w0] = hinge_loss(std_traindata, trainlabel)
# print("hinge loss",w_hi)
[w_hin_reg, w0] = hinge_loss_regularized(std_traindata, trainlabel)
# print("regularised hinge",w_hin_reg)
[w_logistic, w0] = logistic_loss(std_traindata, trainlabel)


###################
## Optional for testing on toy data
## Comment out when submitting project
###################
# print(w)
# wlen = math.sqrt(w[0]**2 + w[1]**2)
# dist_to_origin = abs(w[2])/wlen
# print("Dist to origin=",dist_to_origin)
#
# wlen=0
# for i in range(0, len(w), 1):
# 	wlen += w[i]**2
# wlen=math.sqrt(wlen)
# print("wlen=",wlen)

###################
#### Classify unlabeled points
##################

def zero_prediction(w):
    pred_result = []
    for i in range(0, test_row, 1):
        dotp = dotproduct(w, std_testdata[i])
        if dotp > 0:
            # print("1", i)
            pred_result.append(1)
        else:
            # print("-1", i)
            pred_result.append(-1)
    return pred_result


def log_prediction(w):
    log_result = []
    for i in range(0, test_row, 1):
        dotp = dotproduct(w, std_testdata[i])
        if dotp > 0.5:
            # print("1", i)
            log_result.append(1)
        else:
            # print("-1", i)
            log_result.append(0)
    return log_result


least_sq = zero_prediction(w_l)
OUT = open("least_squares_predictions", 'w')
OUT.write(str(least_sq))
OUT.close()
leastreg_sq = zero_prediction(w_lr)
OUT = open("least_square_regularised_predictions", 'w')
OUT.write(str(leastreg_sq))
OUT.close()
hinge_loss = zero_prediction(w_hi)
OUT = open("hinge_loss_predictions", 'w')
OUT.write(str(hinge_loss))
OUT.close()
OUT = open("hinge_loss_regularised", 'w')
hinge_reg = zero_prediction(w_hin_reg)
OUT.write(str(hinge_reg))
OUT.close()
OUT = open("logistic_regression",'w')
logistic_reg = log_prediction(w_logistic)
OUT.write(str(logistic_reg))
OUT.close()
OUT = open("hinge_adaptive_prediction",'w')
hinge_adaptive = zero_prediction(w_hinge_Adaptive)
OUT.write(str(hinge_adaptive))
OUT.close()
# OUT.write(str(classify()))

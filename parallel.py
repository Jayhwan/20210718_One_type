from common_function import *

load = np.load("one_type_load.npy", allow_pickle=True)

a_o_init = np.ones((2, total_user, time_step))

for i in range(20, 40):
    print("active user number :", i)
    exp_i = np.load("exp_shared_identical_price_"+str(i)+".npy", allow_pickle=True)
    exp_i_rand = np.load("exp_shared_random_identical_price_"+str(i)+".npy", allow_pickle=True)
    x = []
    for j in range(len(exp_i)):
        x += [exp_i[j]]
    for j in range(len(exp_i), len(exp_i_rand)):
        print(len(x))
        if i == 0:
            a_o = None
            a_f = None
        else:
            [a_o, _, _] = exp_i_rand[j]
            #a_o = 1 * np.random.random((2, time_step))+ 0.5 * np.ones((2, time_step))
            #tmp = np.minimum(a_o[0], a_o[1])
            #a_o[1] = np.maximum(a_o[0], a_o[1])
            #a_o[0] = tmp
            a_o, a_f = iterations(i, time_step, load[:total_user, :time_step], a_o)
        x += [[a_o, a_f, load[:total_user, :time_step]]]
        np.save("exp_shared_identical_price_"+str(i)+".npy", x)

from common_function import *
from indiv_function import *

load = np.load("one_type_load.npy", allow_pickle=True)

a_o_init = np.ones((2, total_user, time_step))

for i in range(51):
    print("active user number :", i)
    exp_i = np.load("exp_individual_identical_price_"+str(i)+".npy", allow_pickle=True)
    x = []
    exp_i_rand = np.load("exp_shared_random_identical_price_"+str(i)+".npy", allow_pickle=True)

    for j in range(len(exp_i)):
        x += [exp_i[j]]
    for j in range(len(exp_i), len(exp_i_rand)):
        print(len(x))
        if i == 0:
            a_o = None
            a_f = None
        else:
            if [i, j] in [[5, 1]]:
                a_o = 1 * np.random.random((2, time_step))+ 0.5 * np.ones((2, time_step))
                tmp = np.minimum(a_o[0], a_o[1])
                a_o[1] = np.maximum(a_o[0], a_o[1])
                a_o[0] = tmp
            else:
                [a_o, _, _] = exp_i_rand[j]
            #a_o = 1 * np.random.random((2, time_step))+ 0.5 * np.ones((2, time_step))
            #tmp = np.minimum(a_o[0], a_o[1])
            #a_o[1] = np.maximum(a_o[0], a_o[1])
            #a_o[0] = tmp
            a_o, a_f = iterations_indiv(i, time_step, load[:total_user, :time_step], a_o)
        x += [[a_o, a_f, load[:total_user, :time_step]]]
        np.save("exp_individual_identical_price_"+str(i)+".npy", x)


def find_good_initial_point_indiv(act_user, t, load_matrix):
    max_scatter = 2
    min_ec = np.inf
    min_par = np.inf
    best_ec_a_o = None
    best_par_a_o = None
    best_ec_index=0
    best_par_index=0
    for i in range(max_scatter):
        a_o = 2 * np.random.random((2, t))+1* np.ones((2, t))
        tmp = np.minimum(a_o[0], a_o[1])
        a_o[1] = np.maximum(a_o[0], a_o[1])
        a_o[0] = tmp

        result, x_s, x_b, l, time = follower_action_indiv(act_user, t, a_o, load_matrix)
        a_f = np.array([x_s, x_b, l])
        if result == - np.inf:
            print("fail")
            continue
        else:
            ec = get_ec(act_user, t, load_matrix, a_o, a_f)
            pa = get_par(act_user, t, load_matrix, a_o, a_f)
            if ec < min_ec:
                best_ec_a_o = a_o
                best_ec_index = i
                min_ec = ec
                print(ec, pa)
            elif pa < min_par:
                best_par_a_o = a_o
                best_par_index = i
                min_par = pa
    print(min_ec, min_par, best_ec_index, best_par_index)
    return best_ec_a_o, best_par_a_o


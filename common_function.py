from parameters import *


def decompose(act_user, time, load_matrix, operator_action, user_action):

    if total_user == act_user:
        load_active = load_matrix
        load_passive = np.zeros((1, time))
    elif act_user == 0:
        load_active = np.zeros((1, time))
        load_passive = load_matrix
    else:
        load_active = load_matrix[:act_user, :]
        load_passive = load_matrix[act_user:, :]

    if user_action is not None:
        [x_s, x_b, l] = [user_action[0], user_action[1], user_action[2]]
    else:
        [x_s, x_b, l] = [np.zeros((1, time)), np.zeros((1, time)), np.zeros((1, time))]

    if operator_action is not None:
        [p_s, p_b] = [operator_action[0], operator_action[1]]
    else:
        [p_s, p_b] = [np.zeros((1, time)), np.zeros((1, time))]

    return [x_s, x_b, l, p_s, p_b, load_active, load_passive]


def get_par(act_user, time, load_matrix, operator_action=None, user_action=None):

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, time, load_matrix, operator_action, user_action)

    load_total = np.sum(l, axis=0) + np.sum(load_passive, axis=0)

    return np.max(load_total)/np.average(load_total)


def get_ec(act_user, time, load_matrix, operator_action=None, user_action=None):

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, time, load_matrix, operator_action, user_action)

    load_total = np.sum(l, axis=0) + np.sum(load_passive, axis=0)

    return p_l * np.sum(np.power(load_total, 2))


def operator_objective(act_user, time, load_matrix, operator_action=None, user_action=None):

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, time, load_matrix, operator_action, user_action)

    ec = - get_ec(act_user, time, load_matrix, operator_action, user_action)
    tax = - p_tax * act_user * np.sum(np.power(p_s, 2) + np.power(p_b, 2))

    return ec + tax


def user_objective(act_user, time, load_matrix, operator_action=None, user_action=None):

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, time, load_matrix, operator_action, user_action)

    if act_user == 0:
        return np.zeros(1)

    x = np.zeros(act_user)

    for i in range(act_user):
        trans_cost = np.sum(np.multiply(p_s, x_s[i]) - np.multiply(p_b, x_b[i]))
        energy_cost = - p_l * np.sum(np.multiply(l[i], np.sum(l, axis=0) + np.sum(load_passive, axis=0)))
        soh_cost = - p_soh * np.sum(np.multiply(x_s[i], np.sum(x_s, axis=0)))
        soh_cost += - p_soh * np.sum(np.multiply(x_b[i], np.sum(x_b, axis=0)))
        x[i] = trans_cost + energy_cost + soh_cost

    return x


def operator_constraint_value_identical(act_user, time, load_matrix, operator_action=None, user_action=None):

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, time, load_matrix, operator_action, user_action)

    if act_user == 0:
        return np.zeros((1, time))

    x = np.zeros((2, time))

    x[0] = - p_s
    x[1] = p_s - p_b

    return x


def follower_constraint_value(act_user, time, load_matrix, operator_action=None, user_action=None):

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, time, load_matrix, operator_action, user_action)

    if act_user == 0:
        return np.zeros((1, time))

    x = np.zeros((4 + 3 * act_user, time))

    for t in range(time):
        if t == 0:
            x[0, t] = - (q_min + beta_s * np.sum(x_s[:, t]) - beta_b * np.sum(x_b[:, t]))
            x[1, t] = - x[0, t] - q_max
        else:
            x[0, t] = alpha * x[0, t-1] - (beta_s * np.sum(x_s[:, t]) - beta_b * np.sum(x_b[:, t]))
            x[1, t] = - x[0, t] - q_max
        x[2, t] = np.sum(x_s[:, t]) - c_max
        x[3, t] = np.sum(x_b[:, t]) - c_min
        for i in range(act_user):
            x[4+3*i, t] = - x_s[i, t]
            x[4+3*i+1, t] = - x_b[i, t]
            x[4+3*i+2, t] = - l[i, t]
    return x


def follower_constraints_derivative_matrix(act_user, time):

    x = np.zeros((4 + 3 * act_user, time, 2, time))

    for t in range(time):
        if t == 0:
            x[0, t, 0, t] = act_user * (beta_s * p_l - beta_b * (p_l + p_soh))
            x[0, t, 1, t] = act_user * (beta_s * (p_l + p_soh) - beta_b * p_l)
            x[1, t] = - x[0, t]
        else:
            x[0, t] = alpha * x[0, t-1]
            x[0, t, 0, t] = act_user * (beta_s * p_l - beta_b * (p_l + p_soh))
            x[0, t, 1, t] = act_user * (beta_s * (p_l + p_soh) - beta_b * p_l)
            x[1, t] = - x[0, t] #########

        x[2, t, 0, t] = - act_user * p_l
        x[2, t, 1, t] = - act_user * (p_l + p_soh)
        x[3, t, 0, t] = - act_user * (p_l + p_soh)
        x[3, t, 1, t] = - act_user * p_l

        for i in range(act_user):
            x[4+3*i, t, 0, t] = p_l
            x[4+3*i, t, 1, t] = p_l + p_soh
            x[4+3*i+1, t, 0, t] = p_l + p_soh
            x[4+3*i+1, t, 1, t] = p_l
            x[4 + 3 * i + 2, t, 0, t] = - p_soh
            x[4 + 3 * i + 2, t, 1, t] = p_soh

    return x/(p_soh*(p_soh+2*p_l))


def almost_same(a, b):
    if np.abs(a-b) <= epsilon:
        return True
    else:
        return False


def direction_finding(act_user, time, load_matrix, operator_action, user_action):

    follower_gradient_matrix = follower_constraints_derivative_matrix(act_user, time)
    follower_constraints_value = follower_constraint_value(act_user, time, load_matrix, operator_action, user_action)

    d = cp.Variable(1)
    r = cp.Variable((2, time))
    v = cp.Variable((2, time))
    g = cp.Variable((4+3*act_user, time))

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, time, load_matrix, operator_action, user_action)

    objective = d

    constraints = []

    load_total = np.sum(l, axis=0) + np.sum(load_passive, axis=0)

    # First constraint
    constraints += [cp.sum(cp.multiply(2*p_tax*act_user*p_s+2*p_l*act_user*load_total/(p_soh+2*p_l), r[0]))
                    + cp.sum(cp.multiply(2*p_tax*act_user*p_b+2*p_l*act_user*load_total/(p_soh+2*p_l), r[1]))
                    + cp.sum(cp.multiply(2*act_user*p_l/(p_soh+2*p_l)*load_total, v[0]))
                    - cp.sum(cp.multiply(2*act_user*p_l/(p_soh+2*p_l)*load_total, v[1])) <= d]

    # Second constraints
    for t in range(time):
        constraints += [-r[0, t] <= p_s[t] + d]
        constraints += [r[0, t] - r[1, t] <= - p_s[t] + p_b[t] + d]

    # Third constraints
    for t in range(time):
        constraints += [2*act_user*act_user/(p_soh*(p_soh+2*p_l))*((p_l+p_soh)*v[0, t]+p_l*v[1, t])
                        - act_user*(2*act_user-1)/(p_soh*(p_soh+2*p_l))*(p_l*r[0, t]-(p_soh+p_l)*r[1, t])
                        + cp.sum(cp.multiply(follower_gradient_matrix[:, :, 0, t], g)) == 0]

        constraints += [2*act_user*act_user/(p_soh*(p_soh+2*p_l))*(p_l*v[0, t]+(p_l+p_soh)*v[1, t])
                        - act_user*(2*act_user-1)/(p_soh*(p_soh+2*p_l))*((p_soh+p_l)*r[0, t] - p_l*r[1, t])
                        + cp.sum(cp.multiply(follower_gradient_matrix[:, :, 1, t], g)) == 0]

    # Fourth constraints
    for i in range(4+3*act_user):
        for t in range(time):
            if almost_same(follower_constraints_value[i, t], 0):
                constraints += [g[i, t] >= 0]
                constraints += [cp.sum(cp.multiply(follower_gradient_matrix[i, t], v)) == 0]
            else:
                constraints += [g[i, t] == 0]
                constraints += [cp.sum(cp.multiply(follower_gradient_matrix[i, t] , v)) <= - follower_constraints_value[i, t] + d]

    # Fifth constraint
    constraints += [cp.sum(cp.power(r, 2)) <= 1]

    prob = cp.Problem(cp.Minimize(objective), constraints)
    result = prob.solve(solver='ECOS', max_iters=1000)

    return result, d.value, r.value, v.value, g.value


def follower_action(act_user, time, operator_action, load_matrix):

    if total_user == act_user:
        load_active = load_matrix
        load_passive = np.zeros((1, time))
    else:
        load_active = load_matrix[:act_user, :]
        load_passive = load_matrix[act_user:, :]

    c1 = cp.Variable((1, time))
    c2 = cp.Variable((1, time))

    c1_mat = c1
    c2_mat = c2

    p_s = operator_action[0].reshape(1, -1)
    p_b = operator_action[1].reshape(1, -1)
    p_s_mat = p_s
    p_b_mat = p_b
    for i in range(act_user -1):
        c1_mat = cp.vstack([c1_mat, c1])
        c2_mat = cp.vstack([c2_mat, c2])
        p_s_mat = np.vstack([p_s_mat, p_s])
        p_b_mat = np.vstack([p_b_mat, p_b])

    x_s = ((p_l + p_soh) * p_s_mat - p_l * p_b_mat - p_l * c1_mat - (p_l + p_soh) * c2_mat - p_l * p_soh * load_active)/(p_soh*(p_soh+2*p_l))
    x_b = (p_l * p_s_mat - (p_l + p_soh) * p_b_mat - (p_l + p_soh) * c1_mat - p_l * c2_mat + p_l * p_soh * load_active)/(p_soh*(p_soh+2*p_l))
    l = x_s - x_b + load_active

    utility = cp.sum(cp.multiply(p_s_mat, x_s) - cp.multiply(p_b_mat, x_b))
    utility += - p_l * cp.sum(cp.power(cp.sum(l, axis=0), 2) + cp.multiply(cp.sum(l, axis=0), np.sum(load_passive, axis=0)))
    utility += - p_soh * cp.sum(cp.power(cp.sum(x_s, axis=0), 2) + cp.power(cp.sum(x_b, axis=0), 2))

    constraints = []
    constraints += [- x_s <= 0]
    constraints += [- x_b <= 0]
    constraints += [- l <= 0]

    ess_matrix = np.fromfunction(np.vectorize(lambda i, j: 0 if i < j else np.power(alpha, i - j)),
                                 (time, time), dtype=float)
    q_ess = q_min * np.fromfunction(np.vectorize(lambda i, j: np.power(alpha, i - j + 1)), (time, 1), dtype=float)\
            + beta_s * ess_matrix @ cp.sum(x_s, axis=0).T - beta_b * ess_matrix @ cp.sum(x_b, axis=0).T

    constraints += [q_min <= q_ess, q_ess <= q_max]
    constraints += [cp.sum(x_s, axis=0) <= c_max]
    constraints += [cp.sum(x_b, axis=0) <= c_min]

    prob = cp.Problem(cp.Maximize(utility), constraints)

    start = timer.time()

    result = prob.solve(solver='ECOS', max_iters=1000)

    end = timer.time()

    return result, x_s.value, x_b.value, l.value, end - start


def step_size(act_user, time, load_matrix, operator_action, user_action, d, r):

    update_coef = 0.5
    s = 0.95
    for i in range(2000):
        next_operator_action = operator_action + s * r
        result, x_s, x_b, l, taken_time = follower_action(act_user, time, next_operator_action, load_matrix)
        next_follower_action = np.array([x_s, x_b, l])
        update = True
        if operator_objective(act_user, time, load_matrix, next_operator_action, next_follower_action)\
            >= operator_objective(act_user, time, load_matrix, operator_action, user_action) - update_coef * s * d:
            if np.any(operator_constraint_value_identical(act_user, time, load_matrix, next_operator_action, next_follower_action) > 0):
                update = False
        else:
            update = False

        if update:
            return s, next_operator_action, next_follower_action
        else:
            s *= update_coef
    return 0, operator_action, user_action


def iterations(act_user, time, load_matrix, operator_action):

    a_o = operator_action
    result, x_s, x_b, l, taken_time = follower_action(act_user, time, a_o, load_matrix)

    a_f = np.array([x_s, x_b, l])

    min_step_size = 1e-10

    print(get_ec(act_user, time, load_matrix, a_o, a_f))
    print(get_par(act_user, time, load_matrix, a_o, a_f))

    for i in range(max_iter):
        print("ITER :", i)
        print("DIRECTION")
        result, d, r, v, g = direction_finding(act_user, time, load_matrix, a_o, a_f)
        print("d :", result)
        #print("r :", r)
        print("STEP")
        b, next_a_o, next_a_f = step_size(act_user, time, load_matrix, a_o, a_f, d, r)
        if b != 0:
            print("s :", b)
            a_o = next_a_o
            a_f = next_a_f
        else:
            print("NO UPDATE")
            return a_o, a_f

        print("EC  :", get_ec(act_user, time, load_matrix, a_o, a_f))
        print("PAR :", get_par(act_user, time, load_matrix, a_o, a_f))
        if b <= min_step_size:
            break

    return a_o, a_f



